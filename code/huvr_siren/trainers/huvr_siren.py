"""Terrain pretrainer with Modulated SIREN decoder.

Uses HuvrSiren (per-patch activation modulation, no global
token) instead of the original Huvr + HypoMLP + Upsampler.

The training loss is the mean-squared elevation error of the paper.
"""

import gc
import os

import einops
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from huvr_siren.data.terrain import check_tile_batch
from huvr_siren.models import HuvrSiren
from huvr_siren.utils import (
    NativeScalerWithGradNormCount,
    adjust_learning_rate,
    make_coord_grid,
)

from .huvr import HuvrTrainer


class HuvrSirenTrainer(HuvrTrainer):

    def make_model(self) -> None:
        cfg = self.cfg
        model = HuvrSiren(
            tokenizer_cfg=cfg["tokenizer"],
            hypo_siren_cfg=cfg["hypo_siren"],
            transformer_encoder_cfg=cfg["transformer_encoder"],
            transformer_decoder_cfg=cfg["transformer_decoder"],
            embedding_dim=cfg["embedding_dim"],
            input_channels=cfg["input_channels"],
            siren_modulation=cfg["siren_modulation"],
            input_proj_mode=cfg["input_proj_mode"],
        )
        if self.is_master:
            model.get_param_counts()

        # Checkpoint loading: --resume > pretrain_path > fresh. Resuming is
        # never implicit: rerunning a config starts a new run.
        ckpt_suffix = cfg["ckpt_suffix"]
        if ckpt_suffix is not None:
            ckpt_path = os.path.join(
                cfg["log_root"],
                f"checkpoints/{cfg['dataset_name']}_{cfg['exp_name']}_{ckpt_suffix}.pth",
            )
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(
                    f"--resume={ckpt_suffix} but checkpoint not found: {ckpt_path}"
                )
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            self.epoch = checkpoint["epoch"]
            self.optimizer_state = checkpoint["optimizer"]
            self.scaler_state = checkpoint["scaler"]
            del checkpoint
            gc.collect()
            torch.cuda.empty_cache()
            if self.is_master:
                print(f"Resumed from {ckpt_path}, epoch {self.epoch}")
        elif cfg["pretrain_path"] is not None:
            checkpoint = torch.load(cfg["pretrain_path"], map_location="cpu")
            msg = model.load_state_dict(checkpoint["model"], strict=False)
            if self.is_master:
                print(f"Loaded pretrain from {cfg['pretrain_path']}: {msg}")
            del checkpoint
            gc.collect()
            torch.cuda.empty_cache()
            self.epoch = 1
            self.optimizer_state = None
            self.scaler_state = None
        else:
            self.epoch = 1
            self.optimizer_state = None
            self.scaler_state = None

        if self.distributed:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model.cuda()
            model_ddp = DistributedDataParallel(
                model, device_ids=[self.rank]
            )
        else:
            model.cuda()
            model_ddp = model
        self.model = model
        self.model_ddp = model_ddp

        self._patch_size = cfg["tokenizer"]["patch_size"]
        self._input_size = cfg["tokenizer"]["input_size"]
        self._n_patches = int((self._input_size / self._patch_size) ** 2)
        self._grid_size = self._patch_size  # full resolution per patch

        # Cache the base coord grid — shape never changes
        self._coords_base = make_coord_grid(
            [self._grid_size, self._grid_size], (-1, 1)
        )

    def _to_model_input(self, elev_norm: torch.Tensor) -> torch.Tensor:
        """SIREN model has input_proj internally, pass 1ch directly."""
        # Normalize [0,1] -> [-1,1]
        return (elev_norm - 0.5) / 0.5

    def _reconstruct(self, model_input: torch.Tensor) -> torch.Tensor:
        """Run forward: encoder -> hypernetwork -> SIREN decoder.

        Returns:
            (B, 1, H, W) reconstructed elevation in model output space.
        """
        B = model_input.shape[0]

        ret_dict = self.model_ddp(model_input)
        hyponet = ret_dict["hyponet"]

        coords = einops.repeat(
            self._coords_base.to(model_input.device),
            "h w d -> (b p) h w d", b=B, p=self._n_patches,
        )

        output = hyponet(coords)  # (B*P, H_patch, W_patch, 1)
        patches = einops.rearrange(output, "(b p) h w c -> b p h w c", b=B)
        p_side = int(np.sqrt(patches.shape[1]))
        output = einops.rearrange(
            patches,
            "b (p1 p2) h w c -> b c (p1 h) (p2 w)",
            p1=p_side,
            p2=p_side,
        )  # (B, 1, H, W)

        return output

    def _iter_step(
        self, cur_step: int, data: dict, is_train: bool
    ) -> dict[str, float]:
        elevation = data["elevation"].to(self.rank)  # (B, 1, H, W)
        B = elevation.shape[0]

        check_tile_batch(elevation, self._input_size)
        elev_norm, z_mins, z_range = self._normalize_batch(elevation)
        model_input = self._to_model_input(elev_norm)

        output = self._reconstruct(model_input)  # (B, 1, H, W)
        output_01 = output * 0.5 + 0.5
        del output

        sq_err = (output_01 - elev_norm) ** 2
        mse_per_sample = sq_err.detach().view(B, -1).mean(dim=-1)
        mse_loss = sq_err.mean()
        del sq_err

        loss = mse_loss

        with torch.no_grad():
            psnr = (-10 * torch.log10(mse_per_sample.clamp(min=1e-10))).mean()

            # RMSE in meters
            pred_meters = output_01 * z_range + z_mins
            rmse_meters = (
                ((pred_meters - elevation) ** 2).view(B, -1).mean(dim=-1)
            ).sqrt().mean()

            max_error = (
                (pred_meters - elevation).abs().view(B, -1).max(dim=-1).values
            ).mean()

        learning_rate = 0.0
        if is_train:
            learning_rate = adjust_learning_rate(
                self.optimizer,
                cur_step + self.train_loader_length * (self.epoch - 1),
                self.cfg["warmup_epochs"] * self.train_loader_length,
                self.cfg["max_epochs"] * self.train_loader_length,
                1e-7,
                self.cfg["lr"]
                * self.cfg["batch_size"]
                * self.tot_gpus
                / 256,
            )

            self.scaler(
                loss,
                self.optimizer,
                clip_grad=self.cfg["clip"],
                parameters=self.model_ddp.parameters(),
                create_graph=False,
                update_grad=True,
            )
            self.optimizer.zero_grad()

        return {
            "loss": loss.item(),
            "mse_loss": mse_loss.item(),
            "psnr": psnr.item(),
            "rmse_m": rmse_meters.item(),
            "max_err_m": max_error.item(),
            "lr": learning_rate,
        }

    def train_epoch(self) -> float:
        """Train one epoch and return average loss (all-reduced across GPUs)."""
        self.model_ddp.train()
        loader_iter = iter(self.train_loader)

        pbar = range(self.train_loader_length)
        if self.is_master:
            pbar = tqdm(pbar, desc="train", leave=False)

        epoch_loss_sum = 0.0
        n_steps = 0
        for cur_step in pbar:
            try:
                data = next(loader_iter)
            except StopIteration:
                loader_iter = iter(self.train_loader)
                data = next(loader_iter)

            ret = self.train_step(cur_step, data)
            epoch_loss_sum += ret["loss"]
            n_steps += 1

            if self.is_master:
                pbar.set_description(
                    f"LR: {ret['lr']:.6f}  "
                    f"PSNR: {ret['psnr']:.1f}  "
                    f"RMSE: {ret['rmse_m']:.2f}m  "
                    f"Loss: {ret['loss']:.6f}"
                )
                wandb.log(ret)

        # All-reduce average loss across GPUs
        avg_loss = epoch_loss_sum / max(n_steps, 1)
        if self.distributed:
            loss_tensor = torch.tensor(avg_loss, device=self.rank)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()

        return avg_loss

    def train(self) -> None:
        cfg = self.cfg

        self.optimizer = torch.optim.AdamW(
            self.model_ddp.parameters(),
            lr=cfg["lr"] * cfg["batch_size"] * self.tot_gpus / 256,
            eps=cfg["opt_eps"],
            weight_decay=cfg["weight_decay"],
        )
        if self.optimizer_state is not None:
            self.optimizer.load_state_dict(self.optimizer_state)

        self.scaler = NativeScalerWithGradNormCount()
        if self.scaler_state is not None:
            self.scaler.load_state_dict(self.scaler_state)

        stop_epoch = cfg["stop_epoch"]

        for epoch in range(self.epoch, stop_epoch + 1):
            self.epoch = epoch
            if self.is_master:
                print(f"\n--- Epoch {epoch}/{cfg['max_epochs']} ---")

            epoch_loss = self.train_epoch()

            if self.is_master:
                print(f"Epoch {epoch} avg loss: {epoch_loss:.6f}")

            # Visualize periodically
            vis_freq = cfg["vis_freq"]
            if self.is_master and (epoch % vis_freq == 0 or epoch == 1):
                vis_data = next(iter(self.train_loader))
                self._visualize_batch(vis_data, epoch)

            if epoch % cfg["ckpt_freq"] == 0:
                self.save_checkpoint(
                    os.path.join(
                        cfg["log_root"],
                        f"checkpoints/{cfg['dataset_name']}"
                        f"_{cfg['exp_name']}_epoch{epoch}.pth",
                    )
                )
                val_metrics = self.validate_epoch()
                if val_metrics and self.is_master:
                    print(
                        f"Epoch {epoch} val: "
                        f"PSNR={val_metrics['val_psnr']:.2f}dB  "
                        f"RMSE={val_metrics['val_rmse_m']:.3f}m",
                        flush=True,
                    )
                    wandb.log({**val_metrics, "epoch": epoch})
            elif epoch == 1:
                val_metrics = self.validate_epoch()
                if val_metrics and self.is_master:
                    print(
                        f"Epoch {epoch} val: "
                        f"PSNR={val_metrics['val_psnr']:.2f}dB  "
                        f"RMSE={val_metrics['val_rmse_m']:.3f}m",
                        flush=True,
                    )
                    wandb.log({**val_metrics, "epoch": epoch})

            self.save_checkpoint(
                os.path.join(
                    cfg["log_root"],
                    f"checkpoints/{cfg['dataset_name']}"
                    f"_{cfg['exp_name']}_latest.pth",
                )
            )
