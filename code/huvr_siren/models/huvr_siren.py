"""Terrain Huvr with Modulated SIREN decoder.

Preserves the HUVR ViT encoder, dimension up/downsample, and decoder
transformer, but uses a Modulated SIREN for terrain elevation reconstruction
with per-patch modulation only (no global token, no weight modulation).

Supports four modulation modes (siren_modulation config):
    'amplitude': h_i = alpha_i . sin(omega_0 * (W_i h + b_i))
    'shift':     h_i = sin(omega_0 * (W_i h + b_i) + phi_i)
    'both':      h_i = alpha_i . sin(omega_0 * (W_i h + b_i) + phi_i)
    'incode':    h_i = exp(a_i) . sin(exp(b_i) * omega_0 * (W_i h + b_i) + c_i) + d_i
"""

import torch
import torch.nn as nn
import einops

from .decoders import SirenDecoder
from .decoders.siren_layers import init_siren_wb
from .hyper_decoder import TransformerEncoder
from .vit_encoder import (
    vitb_8_norm_only_rope_fixed,
    vitb_16_norm_only_rope_fixed,
    vitb_32_norm_only_rope_fixed,
    vitl_16_norm_only_rope_fixed,
)


class HuvrSiren(nn.Module):
    """HUVR-style hypernetwork with Modulated SIREN for terrain elevation.

    Architecture:
        DEM tile (1, H, W) -> input_proj(1->3) -> ViT encoder -> dim down
        -> dim up -> decoder transformer
        -> per-patch modulation postfc -> alpha vectors
        -> Modulated SIREN -> elevation patches -> (1, H, W)
    """

    def __init__(
        self,
        tokenizer_cfg: dict,
        hypo_siren_cfg: dict,
        transformer_encoder_cfg: dict,
        transformer_decoder_cfg: dict,
        embedding_dim: int,
        input_channels: int = 1,
        input_proj_mode: str = 'learned_1x1',
        siren_modulation: str = 'amplitude',
    ):
        super().__init__()

        # --- Input projection: 1-channel DEM -> 3-channel for ViT ---
        if input_proj_mode != 'learned_1x1':
            raise ValueError(f"unknown input_proj_mode={input_proj_mode!r}")
        self.input_proj = nn.Conv2d(input_channels, 3, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        # --- ViT Encoder ---
        encoder_name = transformer_encoder_cfg['name']
        if encoder_name == 'vitb_16_norm_only_rope_fixed':
            self.transformer_encoder = vitb_16_norm_only_rope_fixed(
                img_size=tokenizer_cfg['input_size'],
                use_registers=transformer_encoder_cfg.get('use_registers', False),
            )
        elif encoder_name == 'vitb_8_norm_only_rope_fixed':
            self.transformer_encoder = vitb_8_norm_only_rope_fixed(
                img_size=tokenizer_cfg['input_size'],
                use_registers=transformer_encoder_cfg.get('use_registers', False),
            )
        elif encoder_name == 'vitb_32_norm_only_rope_fixed':
            self.transformer_encoder = vitb_32_norm_only_rope_fixed(
                img_size=tokenizer_cfg['input_size'],
                use_registers=transformer_encoder_cfg.get('use_registers', False),
            )
        elif encoder_name == 'vitl_16_norm_only_rope_fixed':
            self.transformer_encoder = vitl_16_norm_only_rope_fixed(
                img_size=tokenizer_cfg['input_size'],
                use_registers=transformer_encoder_cfg.get('use_registers', False),
            )
        else:
            raise ValueError(f"Unknown encoder: {encoder_name}")

        encoder_dim = transformer_encoder_cfg['dim']

        self.transformer_decoder = TransformerEncoder(**transformer_decoder_cfg)
        decoder_dim = transformer_decoder_cfg['dim']

        # --- Dimension projection ---
        if embedding_dim != encoder_dim:
            self.dimension_downsample = nn.Sequential(
                nn.LayerNorm(encoder_dim),
                nn.Linear(encoder_dim, embedding_dim),
            )
        else:
            self.dimension_downsample = nn.Identity()

        if embedding_dim != decoder_dim:
            self.dimension_upsample = nn.Sequential(
                nn.LayerNorm(embedding_dim),
                nn.Linear(embedding_dim, decoder_dim),
            )
        else:
            self.dimension_upsample = nn.Identity()

        self.hyponet = SirenDecoder(**hypo_siren_cfg)
        omega_0 = hypo_siren_cfg.get('omega_0', 10.0)

        # Base SIREN weights, shared across tiles and repeated per patch in forward.
        self.base_params = nn.ParameterDict()
        for name, shape in self.hyponet.param_shapes.items():
            layer_idx = int(name.replace('wb', ''))
            is_first = (layer_idx == 0)
            wb = init_siren_wb(shape, omega_0, is_first)
            self.base_params[name] = nn.Parameter(wb)

        # siren_modulation: 'amplitude' | 'shift' | 'both' | 'incode'
        valid_modes = {'amplitude', 'shift', 'both', 'incode'}
        if siren_modulation not in valid_modes:
            raise ValueError(
                f"siren_modulation must be one of {valid_modes}, "
                f"got '{siren_modulation}'"
            )
        self.siren_modulation = siren_modulation
        self.use_amplitude = siren_modulation in ('amplitude', 'both')
        self.use_shift = siren_modulation in ('shift', 'both')
        self.use_incode = siren_modulation == 'incode'
        self.hyponet.set_incode_mode(self.use_incode)

        def _make_postfc(out_dim: int, bias_init: float = 0.0) -> nn.Sequential:
            postfc = nn.Sequential(
                nn.LayerNorm(decoder_dim),
                nn.Linear(decoder_dim, out_dim),
            )
            nn.init.zeros_(postfc[1].weight)
            nn.init.constant_(postfc[1].bias, bias_init)
            return postfc

        # Amplitude postfc: output ~1.0 (identity scaling)
        self.mod_postfc = nn.ModuleDict()
        if self.use_amplitude:
            for name, shape in self.hyponet.modulation_shapes.items():
                self.mod_postfc[f'{name}_postfc'] = _make_postfc(
                    shape[0], bias_init=1.0,
                )

        # Shift postfc: output ~0.0 (no phase shift)
        self.shift_postfc = nn.ModuleDict()
        if self.use_shift:
            for name, shape in self.hyponet.shift_shapes.items():
                self.shift_postfc[f'{name}_postfc'] = _make_postfc(shape[0])

        # INCODE shared-trunk postfc: one Linear -> 4*hidden_dim, then chunk
        # All zeros init: exp(0)=1 for amp/freq (identity), 0 for phase/dc
        self.incode_postfc = nn.ModuleDict()
        if self.use_incode:
            for name, shape in self.hyponet.modulation_shapes.items():
                layer_idx = name.replace('mod', '')
                self.incode_postfc[f'incode{layer_idx}_postfc'] = _make_postfc(
                    4 * shape[0],
                )

    def get_param_counts(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        encoder_params = sum(
            p.numel() for p in self.transformer_encoder.parameters()
        )
        mod_params = sum(p.numel() for p in self.mod_postfc.parameters()) + \
            sum(p.numel() for p in self.shift_postfc.parameters()) + \
            sum(p.numel() for p in self.incode_postfc.parameters())
        base_params = sum(p.numel() for p in self.base_params.values())
        proj_params = sum(p.numel() for p in self.input_proj.parameters())

        print(f'Total params: {total:,}')
        print(f'Encoder params: {encoder_params:,}')
        print(f'Modulation postfc params: {mod_params:,}')
        print(f'SIREN base params: {base_params:,}')
        print(f'Input projection params: {proj_params:,}')
        dec_params = sum(p.numel() for p in self.transformer_decoder.parameters())
        print(f'Decoder transformer params: {dec_params:,}')

    def forward(self, data: torch.Tensor) -> dict:
        """(B, 1, H, W) elevation tile -> the SIREN decoder, modulated per patch."""
        x = self.input_proj(data)
        trans_enc_out = self.transformer_encoder(x)

        # Per-patch bottleneck: (B, 1+P, encoder_dim) -> (B, 1+P, embedding_dim)
        z = self.dimension_downsample(trans_enc_out)

        trans_out = self.transformer_decoder(self.dimension_upsample(z))

        # Extract patch tokens — always skip CLS at position 0
        patch_out = trans_out[:, 1:, :]
        B, P = patch_out.shape[0], patch_out.shape[1]

        params: dict[str, torch.Tensor] = {}
        for name in self.hyponet.param_shapes:
            params[name] = self.base_params[name].unsqueeze(0).expand(
                B * P, -1, -1
            )

        modulations: dict[str, torch.Tensor] = {}
        if self.use_amplitude:
            for name in self.hyponet.modulation_shapes:
                alpha = self.mod_postfc[f'{name}_postfc'](patch_out)
                modulations[name] = einops.rearrange(
                    alpha, 'b p d -> (b p) 1 d'
                )

        shifts: dict[str, torch.Tensor] = {}
        if self.use_shift:
            for name in self.hyponet.shift_shapes:
                phi = self.shift_postfc[f'{name}_postfc'](patch_out)
                shifts[name] = einops.rearrange(phi, 'b p d -> (b p) 1 d')

        frequencies: dict[str, torch.Tensor] = {}
        dc_offsets: dict[str, torch.Tensor] = {}
        if self.use_incode:
            for name in self.hyponet.modulation_shapes:
                layer_idx = name.replace('mod', '')
                raw = self.incode_postfc[f'incode{layer_idx}_postfc'](
                    patch_out
                )
                raw = einops.rearrange(raw, 'b p (n d) -> n (b p) 1 d', n=4)
                modulations[name] = raw[0]
                frequencies[f'freq{layer_idx}'] = raw[1]
                shifts[f'shift{layer_idx}'] = raw[2]
                dc_offsets[f'dc{layer_idx}'] = raw[3]

        self.hyponet.set_params(params)
        self.hyponet.set_modulations(modulations)
        self.hyponet.set_shifts(shifts)
        self.hyponet.set_frequencies(frequencies)
        self.hyponet.set_dc_offsets(dc_offsets)

        return {'hyponet': self.hyponet}
