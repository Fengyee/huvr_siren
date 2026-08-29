"""Terrain pretrainer using the original HuVR Huvr architecture.

Treats elevation patches as grayscale images: 1ch → repeat 3ch for input,
3ch output → mean to 1ch for loss. No architecture modifications.
"""

import gc
import math
import os

import einops
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from huvr_siren.data.terrain import (
    check_tile_batch,
    create_terrain_dataloader,
    create_terrain_subset_dataloader,
)
from huvr_siren.models import Huvr
from huvr_siren.utils import (
    NativeScalerWithGradNormCount,
    adjust_learning_rate,
    make_coord_grid,
)

from .base import Trainer


def finite_diff_gradient(elevation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute spatial gradients via central finite differences.

    Args:
        elevation: (H, W) elevation in meters.

    Returns:
        (dz_dx, dz_dy) each (H, W), in meters/pixel.
    """
    dz_dy = np.zeros_like(elevation)
    dz_dx = np.zeros_like(elevation)
    dz_dy[1:-1, :] = (elevation[2:, :] - elevation[:-2, :]) / 2.0
    dz_dx[:, 1:-1] = (elevation[:, 2:] - elevation[:, :-2]) / 2.0
    dz_dy[0, :] = elevation[1, :] - elevation[0, :]
    dz_dy[-1, :] = elevation[-1, :] - elevation[-2, :]
    dz_dx[:, 0] = elevation[:, 1] - elevation[:, 0]
    dz_dx[:, -1] = elevation[:, -1] - elevation[:, -2]
    return dz_dx, dz_dy


def aggregate_val_metrics(per_batch: list[dict]) -> dict[str, float]:
    """Sample-weighted aggregation of per-batch val metrics.

    Per-tile PSNRs are averaged in dB within a batch, so combining batches by
    converting back to a linear error and re-expressing in dB does not give the
    PSNR of the pooled MSE. This is a training monitor, not a benchmark number.

    Args:
        per_batch: list of {"psnr": float, "rmse_m": float, "bsz": int}.
    """
    if not per_batch:
        raise ValueError("Cannot aggregate val metrics from empty list")

    total_samples = sum(b["bsz"] for b in per_batch)
    weighted_mse = sum(
        (10.0 ** (-b["psnr"] / 10.0)) * b["bsz"] for b in per_batch
    )
    mean_mse = weighted_mse / total_samples
    mean_rmse = sum(b["rmse_m"] * b["bsz"] for b in per_batch) / total_samples
    return {
        "val_psnr": -10.0 * math.log10(max(mean_mse, 1e-10)),
        "val_rmse_m": mean_rmse,
    }


class HuvrTrainer(Trainer):

    def __init__(self, rank, cfg):
        super().__init__(rank, cfg)
        if self.is_master:
            wandb.init(
                project=cfg["wandb_project"],
                name=f"{cfg['dataset_name']}_{cfg['exp_name']}_terrain",
            )
            wandb.config.update(cfg)

    def make_datasets(self):
        cfg = self.cfg
        data_dir = cfg["terrain_data_dir"]
        subset_n = cfg["terrain_subset_n"]
        if subset_n is not None and cfg["terrain_augment"]:
            raise ValueError(
                "terrain_augment is not applied in subset mode "
                "(terrain_subset_n is set); set terrain_augment: false."
            )

        if subset_n is not None:
            # Fixed subset mode: load exactly N samples, no augmentation.
            # terrain_subset_seed enables nested reproducible selection
            # (smaller N ⊂ larger N under the same seed).
            # Passing rank/world_size distributes the subset across DDP
            # ranks so each GPU sees a disjoint slice per step.
            self.train_loader, subset_meta = create_terrain_subset_dataloader(
                data_dir=os.path.join(data_dir, "train"),
                n=subset_n,
                batch_size=cfg["batch_size"],
                seed=cfg["terrain_subset_seed"],
                rank=self.rank if self.distributed else 0,
                world_size=self.tot_gpus if self.distributed else 1,
            )
            if self.is_master:
                print(f"Subset mode: {subset_n} fixed samples")
                for i, m in enumerate(subset_meta):
                    print(
                        f"  [{i}] tile={m.get('tile_id','?')} "
                        f"crop={m.get('crop_xy','?')} "
                        f"mu={m.get('mu',0):.1f} sigma={m.get('sigma',0):.2f}"
                    )
                # Save subset info to experiment dir
                import json
                info_dir = os.path.join(
                    cfg["log_root"],
                    "checkpoints",
                )
                os.makedirs(info_dir, exist_ok=True)
                info_path = os.path.join(info_dir, "subset_info.json")
                with open(info_path, "w") as f:
                    json.dump(subset_meta, f, indent=2)
                print(f"Subset info saved to {info_path}")
        else:
            self.train_loader = create_terrain_dataloader(
                data_dir=os.path.join(data_dir, "train"),
                batch_size=cfg["batch_size"],
                shuffle=True,
                num_workers=cfg["num_workers"],
                distributed=self.distributed,
                augment=cfg["terrain_augment"],
            )
        # WebDataset is iterable, so the epoch length comes from the configured
        # sample count. In subset mode this sets how often the subset repeats.
        num_train = cfg["terrain_num_train"]
        self.train_loader_length = max(
            1, num_train // (cfg["batch_size"] * self.tot_gpus)
        )
        if self.is_master:
            print(
                f"Train: {num_train} patches, "
                f"{self.train_loader_length} steps/epoch"
            )

        # Val loader: full terrain val split, no shuffle, no augment.
        # Used by validate_epoch() for held-out PSNR/RMSE metrics.
        # IMPORTANT: distributed=False even under DDP. The val split has
        # 1 shard; with split_by_node only rank 0 would receive data and
        # ranks 1-3 would deadlock on the sample-count all-reduce while
        # rank 0 iterates 441 patches. Replicating the full split to
        # every rank lets each rank compute the same metrics locally
        # (no cross-rank communication required).
        val_dir = os.path.join(data_dir, "val")
        if os.path.isdir(val_dir) and cfg["enable_val"]:
            self.val_loader = create_terrain_dataloader(
                data_dir=val_dir,
                batch_size=cfg["batch_size"],
                shuffle=False,
                num_workers=cfg["num_workers"],
                distributed=False,
                replicate=self.distributed,
                augment=False,
            )
            self._val_num_samples = cfg["terrain_num_val"]
            if self.is_master:
                print(
                    f"Val: {self._val_num_samples} patches at {val_dir}"
                )
        else:
            self.val_loader = None
            self._val_num_samples = 0

    def make_model(self):
        cfg = self.cfg
        model = Huvr(
            cfg["tokenizer"],
            cfg["hyponet"],
            cfg["hypocnn"],
            cfg["transformer_encoder"],
            cfg["transformer_decoder"],
            cfg["embedding_dim"],
            cfg["mod_idxs"],
            cfg["use_hypocnn"],
            cfg["use_global_token"],
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

        # Pre-compute decode geometry (static for lifetime of trainer)
        hypo_strides = math.prod(
            int(s) for s in cfg["hyponet"]["strides"].split("_")
        )
        cnn_strides = (
            math.prod(int(s) for s in cfg["hypocnn"]["strds"].split("_"))
            if cfg["use_hypocnn"]
            else 1
        )
        self._upsample_factor = hypo_strides * cnn_strides
        self._patch_size = cfg["tokenizer"]["patch_size"]
        self._input_size = cfg["tokenizer"]["input_size"]
        self._n_patches = int((self._input_size / self._patch_size) ** 2)
        self._grid_size = self._patch_size // self._upsample_factor

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalize_batch(self, elevation):
        """Per-tile min-max normalise elevation to [0, 1].

        Recomputed from the tile itself, so a tile needs no accompanying
        statistics. `z_mins` and `z_range` are returned for the metre-space
        RMSE and for decoding back to elevation.
        """
        B = elevation.shape[0]
        z_mins = elevation.view(B, -1).amin(dim=1).view(B, 1, 1, 1)
        z_maxs = elevation.view(B, -1).amax(dim=1).view(B, 1, 1, 1)
        z_range = (z_maxs - z_mins).clamp(min=1e-6)
        elev_norm = (elevation - z_mins) / z_range
        return elev_norm, z_mins, z_range

    def _to_model_input(self, elev_norm):
        """Convert normalized [0,1] 1ch elevation to [-1,1] 3ch for model."""
        input_3ch = elev_norm.expand(-1, 3, -1, -1)  # (B, 3, H, W)
        return (input_3ch - 0.5) / 0.5  # [-1, 1]

    def _reconstruct(self, model_input):
        """Run full forward pass: encoder → hypernetwork → decoder."""
        B = model_input.shape[0]

        ret_dict = self.model_ddp(model_input)
        hyponet = ret_dict["hyponet"]
        hypocnn = ret_dict.get("hypocnn")

        coords = make_coord_grid(
            [self._grid_size, self._grid_size],
            (-1, 1),
            device=model_input.device,
        )
        coords = einops.repeat(
            coords, "h w d -> (b p) h w d", b=B, p=self._n_patches
        )

        output = hyponet(coords)
        patches = einops.rearrange(output, "(b p) h w c -> b p h w c", b=B)
        p_side = int(np.sqrt(patches.shape[1]))
        output = einops.rearrange(
            patches,
            "b (p1 p2) h w c -> b (p1 h) (p2 w) c",
            p1=p_side,
            p2=p_side,
        )

        if hypocnn is not None:
            output = einops.rearrange(output, "b h w c -> b c h w")
            output = hypocnn(output)
        else:
            output = einops.rearrange(
                output, "b h w c -> b c h w"
            ).contiguous()

        return output  # (B, 3, H, W) in [-1, 1] when normalize_images

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _iter_step(self, cur_step, data, is_train):
        elevation = data["elevation"].to(self.rank)  # (B, 1, H, W)
        B = elevation.shape[0]

        # Normalize elevation to [0, 1], then convert to model input
        check_tile_batch(elevation, self._input_size)
        elev_norm, z_mins, z_range = self._normalize_batch(elevation)
        model_input = self._to_model_input(elev_norm)

        # Forward
        output_3ch = self._reconstruct(model_input)  # (B, 3, H, W)

        # Convert output back: denormalize [-1,1] → [0,1], mean over 3ch
        output_01 = output_3ch * 0.5 + 0.5
        del output_3ch
        output_1ch = output_01.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        del output_01

        # Loss and metrics from single squared-error computation
        sq_err = (output_1ch - elev_norm) ** 2
        mse_per_sample = sq_err.detach().view(B, -1).mean(dim=-1)
        loss = sq_err.mean()
        del sq_err

        with torch.no_grad():
            psnr = (-10 * torch.log10(mse_per_sample.clamp(min=1e-10))).mean()

            # RMSE in meters
            pred_meters = output_1ch * z_range + z_mins
            rmse_meters = (
                ((pred_meters - elevation) ** 2).view(B, -1).mean(dim=-1)
            ).sqrt().mean()

            # Max error in meters
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
                self._opt_base_lr,
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
            "psnr": psnr.item(),
            "rmse_m": rmse_meters.item(),
            "max_err_m": max_error.item(),
            "lr": learning_rate,
        }

    def train_epoch(self):
        self.model_ddp.train()
        loader_iter = iter(self.train_loader)

        pbar = range(self.train_loader_length)
        if self.is_master:
            pbar = tqdm(pbar, desc="train", leave=False)

        for cur_step in pbar:
            try:
                data = next(loader_iter)
            except StopIteration:
                loader_iter = iter(self.train_loader)
                data = next(loader_iter)

            ret = self.train_step(cur_step, data)

            if self.is_master:
                pbar.set_description(
                    f"LR: {ret['lr']:.6f}  "
                    f"PSNR: {ret['psnr']:.1f}  "
                    f"RMSE: {ret['rmse_m']:.2f}m  "
                    f"Loss: {ret['loss']:.6f}"
                )
                wandb.log(ret)

    @torch.no_grad()
    def validate_epoch(self) -> dict:
        """Run one pass over the val loader and return PSNR/RMSE.

        Every DDP rank holds the full val loader (see make_datasets)
        and computes the same metrics independently — no cross-rank
        communication needed. We invert each batch's mean PSNR to MSE,
        sample-weight, then map back to global PSNR via
        ``aggregate_val_metrics``.

        Returns {} if no val loader is configured.
        """
        if self.val_loader is None:
            return {}

        self.model_ddp.eval()
        per_batch: list[dict] = []
        for data in self.val_loader:
            ret = self._iter_step(cur_step=0, data=data, is_train=False)
            per_batch.append({
                "psnr": ret["psnr"],
                "rmse_m": ret["rmse_m"],
                "bsz": data["elevation"].shape[0],
            })

        self.model_ddp.train()

        if not per_batch:
            return {}
        return aggregate_val_metrics(per_batch)

    @torch.no_grad()
    def _visualize_batch(self, data, epoch):
        """Generate GT/pred/error + gradient comparison, log to WandB."""
        self.model_ddp.eval()

        elevation = data["elevation"].to(self.rank)
        metadata = data["metadata"]
        B = elevation.shape[0]
        n_vis = min(B, 4)

        check_tile_batch(elevation, self._input_size)
        elev_norm, z_mins, z_range = self._normalize_batch(elevation)
        model_input = self._to_model_input(elev_norm)
        output_3ch = self._reconstruct(model_input)
        output_01 = output_3ch * 0.5 + 0.5
        output_1ch = output_01.mean(dim=1, keepdim=True)
        pred_meters = output_1ch * z_range + z_mins

        figs = []
        for i in range(n_vis):
            gt = elevation[i, 0].cpu().numpy()
            pr = pred_meters[i, 0].cpu().numpy()
            err = np.abs(pr - gt)
            tile = metadata[i].get("tile_id", f"sample_{i}")

            # Gradients
            gt_dx, gt_dy = finite_diff_gradient(gt)
            pr_dx, pr_dy = finite_diff_gradient(pr)
            gt_grad_mag = np.sqrt(gt_dx ** 2 + gt_dy ** 2)
            pr_grad_mag = np.sqrt(pr_dx ** 2 + pr_dy ** 2)
            grad_err = np.abs(pr_grad_mag - gt_grad_mag)

            # Metrics
            rmse = float(np.sqrt((err ** 2).mean()))
            mse_n = float(((output_1ch[i, 0].cpu().numpy()
                            - elev_norm[i, 0].cpu().numpy()) ** 2).mean())
            psnr = -10 * np.log10(max(mse_n, 1e-10))
            grad_rmse = float(np.sqrt(
                ((pr_dx - gt_dx) ** 2 + (pr_dy - gt_dy) ** 2).mean()
            ))

            fig, axes = plt.subplots(2, 3, figsize=(16, 10))

            # Row 1: elevation
            im0 = axes[0, 0].imshow(gt, cmap="terrain")
            axes[0, 0].set_title(f"GT ({tile})")
            plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

            im1 = axes[0, 1].imshow(pr, cmap="terrain",
                                     vmin=gt.min(), vmax=gt.max())
            axes[0, 1].set_title("Prediction")
            plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

            im2 = axes[0, 2].imshow(err, cmap="hot")
            axes[0, 2].set_title(f"|Error| (max={err.max():.2f}m)")
            plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

            # Row 2: gradient magnitude
            vmax_g = np.percentile(gt_grad_mag, 99)
            im3 = axes[1, 0].imshow(gt_grad_mag, cmap="viridis",
                                     vmin=0, vmax=vmax_g)
            axes[1, 0].set_title("GT Gradient")
            plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

            im4 = axes[1, 1].imshow(pr_grad_mag, cmap="viridis",
                                     vmin=0, vmax=vmax_g)
            axes[1, 1].set_title("Pred Gradient")
            plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

            im5 = axes[1, 2].imshow(grad_err, cmap="hot")
            axes[1, 2].set_title(f"|Grad Error| (RMSE={grad_rmse:.3f})")
            plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)

            for ax in axes.flat:
                ax.axis("off")

            fig.suptitle(
                f"Epoch {epoch}  PSNR={psnr:.1f}dB  "
                f"RMSE={rmse:.3f}m  GradRMSE={grad_rmse:.3f}m/px",
                fontsize=12,
            )
            plt.tight_layout()
            figs.append(wandb.Image(fig, caption=f"{tile}_epoch{epoch}"))
            plt.close(fig)

        # commit=True so the images are finalized as their own step rather than
        # buffered into the next training iter's log (that deferral loses the
        # images in wandb 0.17+ once the step advances). `epoch` gives the panel
        # a clean x-axis.
        wandb.log({"reconstructions": figs, "epoch": epoch})
        self.model_ddp.train()

    def train(self):
        cfg = self.cfg

        opt_lr = cfg["lr"] * cfg["batch_size"] * self.tot_gpus / 256
        self.optimizer = torch.optim.AdamW(
            self.model_ddp.parameters(),
            lr=opt_lr,
            eps=cfg["opt_eps"],
            weight_decay=cfg["weight_decay"],
        )
        self._opt_base_lr = opt_lr
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

            self.train_epoch()

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

    def save_checkpoint(self, filename):
        if not self.is_master:
            return
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        print(f"Saving checkpoint to {filename}", flush=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "epoch": self.epoch + 1,
                "cfg": self.cfg,
            },
            filename,
        )
