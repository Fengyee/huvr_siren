"""Build both models from the shipped configs and decode a synthetic tile.

Needs no data and no GPU:

    python smoke_test.py
"""

from __future__ import annotations

import pathlib

import einops
import torch

from huvr_siren.data.terrain import check_tile_batch
from huvr_siren.models import Huvr, HuvrSiren
from huvr_siren.utils import load_config, make_coord_grid

CONFIGS = pathlib.Path(__file__).parent / "configs"


def synthetic_tile(n: int = 256, seed: int = 0) -> torch.Tensor:
    """A smooth ridge plus noise, in metres — shaped like a real DEM tile."""
    g = torch.linspace(-1, 1, n)
    y, x = torch.meshgrid(g, g, indexing="ij")
    surface = 400 + 120 * torch.exp(-(x**2 + 3 * y**2)) + 30 * torch.sin(4 * x)
    noise = torch.Generator().manual_seed(seed)
    return (surface + 0.5 * torch.randn(n, n, generator=noise))[None, None]


def normalise(tile: torch.Tensor) -> torch.Tensor:
    b = tile.shape[0]
    lo = tile.view(b, -1).amin(1).view(b, 1, 1, 1)
    hi = tile.view(b, -1).amax(1).view(b, 1, 1, 1)
    return (tile - lo) / (hi - lo).clamp(min=1e-6)


def run_huvr_siren(tile01: torch.Tensor) -> torch.Tensor:
    cfg = load_config(CONFIGS / "huvr_siren.yaml", "huvr_siren")
    model = HuvrSiren(
        tokenizer_cfg=cfg["tokenizer"],
        hypo_siren_cfg=cfg["hypo_siren"],
        transformer_encoder_cfg=cfg["transformer_encoder"],
        transformer_decoder_cfg=cfg["transformer_decoder"],
        embedding_dim=cfg["embedding_dim"],
        input_channels=cfg["input_channels"],
        siren_modulation=cfg["siren_modulation"],
        input_proj_mode=cfg["input_proj_mode"],
    ).eval()

    p = cfg["tokenizer"]["patch_size"]
    n_patches = (cfg["tokenizer"]["input_size"] // p) ** 2
    with torch.no_grad():
        decoder = model((tile01 - 0.5) / 0.5)["hyponet"]
        coords = einops.repeat(
            make_coord_grid([p, p], (-1, 1)), "h w d -> (b q) h w d", b=1, q=n_patches
        )
        patches = decoder(coords)
    side = int(n_patches**0.5)
    return einops.rearrange(
        patches, "(b p1 p2) h w c -> b c (p1 h) (p2 w)", b=1, p1=side, p2=side
    ) * 0.5 + 0.5


def run_huvr(tile01: torch.Tensor) -> None:
    cfg = load_config(CONFIGS / "huvr.yaml", "huvr")
    model = Huvr(
        cfg["tokenizer"], cfg["hyponet"], cfg["hypocnn"],
        cfg["transformer_encoder"], cfg["transformer_decoder"],
        cfg["embedding_dim"], cfg["mod_idxs"],
        cfg["use_hypocnn"], cfg["use_global_token"],
    ).eval()
    with torch.no_grad():
        out = model(((tile01 - 0.5) / 0.5).expand(-1, 3, -1, -1))
    assert out["hyponet"] is not None and out["hypocnn"] is not None
    print(f"Huvr        {sum(p.numel() for p in model.parameters()):,} parameters")


def main() -> None:
    torch.manual_seed(0)
    tile = synthetic_tile()
    tile01 = normalise(tile)

    run_huvr(tile01)

    for bad in (torch.zeros(1, 1, 256, 128), torch.zeros(1, 3, 256, 256)):
        try:
            check_tile_batch(bad, 256)
        except ValueError:
            pass
        else:
            raise AssertionError(f"tile of shape {tuple(bad.shape)} was not rejected")

    recon = run_huvr_siren(tile01)
    assert recon.shape == tile01.shape, f"{recon.shape} != {tile01.shape}"
    mse = ((recon - tile01) ** 2).mean()
    print(f"HuvrSiren   reconstruction {tuple(recon.shape)}, "
          f"untrained MSE {mse.item():.4f}")
    print("ok")


if __name__ == "__main__":
    main()
