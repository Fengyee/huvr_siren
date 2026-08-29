### inspired by and some code from trans-inr, see https://github.com/yinboc/trans-inr/blob/f4bdc013286e2be00f9117e4e53913d6692fa49d/models/trans_inr.py

import math

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoders import ReluMlpDecoder, Upsampler
from .hyper_decoder import TransformerEncoder
from .vit_encoder import vitb_16_norm_only_rope_fixed, vitl_16_norm_only_rope_fixed

ENCODERS = {
    'vitb_16_norm_only_rope_fixed': vitb_16_norm_only_rope_fixed,
    'vitl_16_norm_only_rope_fixed': vitl_16_norm_only_rope_fixed,
}


def init_wb(shape):
    weight = torch.empty(shape[1], shape[0] - 1)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

    bias = torch.empty(shape[1], 1)
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    nn.init.uniform_(bias, -bound, bound)

    return torch.cat([weight, bias], dim=1).t().detach()


def init_wb_cnn(shape):
    out_ch, in_ch, ks = shape
    weight = torch.empty(in_ch, out_ch, ks, ks)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

    bias = torch.empty(out_ch)
    _, fan_in = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    nn.init.uniform_(bias, -bound, bound)

    wb_list = [weight.permute(0, 2, 3, 1).flatten(end_dim=-2), bias[None]]
    return torch.cat(wb_list, dim=0).detach()


class Huvr(nn.Module):
    """HUVR as published, ported to single-channel terrain heightfields.

    ViT encoder -> per-patch bottleneck of width ``embedding_dim`` -> hypernetwork
    decoder -> per-patch modulations of a shared ReLU MLP -> Conv+PixelShuffle
    upsampler. The tile enters as three identical channels (see README, "What the
    released code does").
    """

    def __init__(self, tokenizer_cfg, hyponet_cfg, hypocnn_cfg,
                 transformer_encoder_cfg, transformer_decoder_cfg, embedding_dim,
                 mod_idxs, use_hypocnn=True, use_global_token=True):
        super().__init__()

        self.transformer_encoder = ENCODERS[transformer_encoder_cfg['name']](
            img_size=tokenizer_cfg['input_size'],
            use_registers=transformer_encoder_cfg['use_registers'],
        )
        encoder_dim = transformer_encoder_cfg['dim']

        self.transformer_decoder = TransformerEncoder(**transformer_decoder_cfg)
        decoder_dim = transformer_decoder_cfg['dim']

        self.hyponet = ReluMlpDecoder(**hyponet_cfg)
        self.hypocnn = Upsampler(**hypocnn_cfg) if use_hypocnn else None
        self.use_global_token = use_global_token

        self.dimension_downsample = (
            nn.Sequential(nn.LayerNorm(encoder_dim), nn.Linear(encoder_dim, embedding_dim))
            if embedding_dim != encoder_dim else nn.Identity()
        )
        self.dimension_upsample = (
            nn.Sequential(nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, decoder_dim))
            if embedding_dim != decoder_dim else nn.Identity()
        )

        self.mod_idxs = [int(x) for x in mod_idxs.split('_')]

        self.base_params = nn.ParameterDict()
        self.wtoken_postfc = nn.ModuleDict()
        for name, shape in self.hyponet.param_shapes.items():
            self.base_params[name] = nn.Parameter(init_wb(shape))
            if int(name.replace('wb', '')) in self.mod_idxs:
                if use_global_token:
                    self.wtoken_postfc[f'{name}_global_postfc'] = nn.Sequential(
                        nn.LayerNorm(decoder_dim),
                        nn.Linear(decoder_dim, shape[1]),
                    )
                self.wtoken_postfc[f'{name}_patch_postfc'] = nn.Sequential(
                    nn.LayerNorm(decoder_dim),
                    nn.Linear(decoder_dim, shape[0] - 1),
                )
                self.patch_token_dim = shape[0] - 1
                self.global_token_dim = shape[1]

        if use_hypocnn:
            for name, shape in self.hypocnn.param_shapes.items():
                self.base_params[name] = nn.Parameter(init_wb_cnn(shape))

    def get_param_counts(self):
        transformer_params = sum(p.numel() for p in self.transformer_encoder.parameters())
        wtoken_postfc_params = sum(p.numel() for p in self.wtoken_postfc.parameters())
        base_params = sum(p.numel() for p in self.base_params.values())
        print(f'Transformer encoder params: {transformer_params}')
        print(f'Huvr Params: {transformer_params + wtoken_postfc_params + base_params}')
        print(f'Base params: {base_params}')
        print(f'Decoder params: {sum(p.numel() for p in self.transformer_decoder.parameters())}')
        print(f'Global token dim: {self.global_token_dim}, patch token dim: {self.patch_token_dim}')

    def forward(self, data):
        B = data.shape[0]
        global_token_offset = 1 if self.use_global_token else 0

        trans_enc_out = self.transformer_encoder(data)
        trans_dec_in = self.dimension_upsample(self.dimension_downsample(trans_enc_out))
        trans_out = self.transformer_decoder(trans_dec_in)

        if self.use_global_token:
            global_out = trans_out[:, :global_token_offset, :]
        patch_out = trans_out[:, global_token_offset:, :]

        params = dict()
        for name in self.hyponet.param_shapes:
            wb = einops.repeat(self.base_params[name], 'n m -> (b p) n m',
                               b=B, p=patch_out.shape[1])
            w, b = wb[:, :-1, :], wb[:, -1:, :]

            if int(name.replace('wb', '')) in self.mod_idxs:
                x = self.wtoken_postfc[f'{name}_patch_postfc'](patch_out)  # B x P x d
                x = x.unsqueeze(-1)  # B x P x d x 1
                if self.use_global_token:
                    c = self.wtoken_postfc[f'{name}_global_postfc'](global_out)  # B x 1 x d
                    c = einops.repeat(c, 'b 1 d -> b p d', p=x.shape[1]).unsqueeze(-2)
                    x = torch.einsum('b p j k, b p k l -> b p j l', x, c)  # B x P x d x d
                else:
                    x = einops.repeat(x, 'b p j 1 -> b p j l', l=self.global_token_dim)
                x = einops.rearrange(x, 'b p j l -> (b p) j l')
                w = F.normalize(w * x, dim=1)
            else:
                w = F.normalize(w, dim=1)

            params[name] = torch.cat([w, b], dim=1)

        self.hyponet.set_params(params)  ### NOTE: we are batching B * p hyponetworks

        if self.hypocnn is not None:
            cnn_params = dict()
            for name in self.hypocnn.param_shapes:
                wb = einops.repeat(self.base_params[name], 'n m -> b n m', b=B)
                w, b = wb[:, :-1, :], wb[:, -1:, :]
                cnn_params[name] = torch.cat([F.normalize(w, dim=1), b], dim=1)
            self.hypocnn.set_params(cnn_params)

        return {'hyponet': self.hyponet, 'hypocnn': self.hypocnn}
