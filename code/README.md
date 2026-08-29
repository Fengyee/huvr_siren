# HUVR+SIREN

An amortized neural representation encodes a terrain tile as a compact per-tile
token, and a shared coordinate decoder reconstructs the heightfield from it.
**HUVR+SIREN** keeps HUVR's ViT encoder, patch-token bottleneck and hypernetwork
decoder, and replaces its piecewise-linear ReLU coordinate decoder with a
modulated SIREN, evaluated directly at each pixel coordinate.

Implements `huvr_siren` and the `huvr` baseline from *Rethinking Amortized
Neural Representations for High-Resolution Terrain Elevation Data*
(SIGSPATIAL '26).

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Smoke test

Builds both models and decodes a synthetic tile. No data, no GPU:

```bash
python smoke_test.py
```

```
Huvr        120,156,048 parameters
HuvrSiren   reconstruction (1, 1, 256, 256), untrained MSE 0.0583
ok
```

## Data

Tiles are read from WebDataset shards. A shard is a `.tar` named
`shard-{:06d}.tar` with two members per sample:

| Member | Contents |
|---|---|
| `{key}.elevation.npy` | elevation in metres, `float32`, shape `(256, 256)` |
| `{key}.metadata.json` | optional; only `tile_id` is read, to caption figures |

Elevation is stored unnormalised, in metres. Tiles must be square and exactly
`tokenizer.input_size` on a side.

Point `--data` at a directory holding `train/` and `val/` subdirectories of
shards, packed into at least as many shards as you have GPUs.

The benchmark in the paper is built from swisstopo's
[swissALTI3D](https://www.swisstopo.admin.ch/en/height-model-swissalti3d)
bare-earth model: each 2000×2000 tile at 0.5 m/pixel is bilinearly downsampled
to 1000×1000 at 1 m/pixel, and the 3×3 grid of non-overlapping 256×256 crops is
taken from its centre (discarding a 116-pixel margin on each side). Splits are
assigned at the source-tile level, so no two crops in different splits overlap
spatially. The benchmark holds 3338 training, 420 validation and 432 test tiles.

## Training

```bash
torchrun --nproc_per_node=4 train.py \
    --method huvr_siren \
    --config configs/huvr_siren.yaml \
    --data /path/to/terrain_shards \
    --log-root runs/
```

Swap `--method huvr --config configs/huvr.yaml` for the baseline.

Metrics go to Weights & Biases, so run `wandb login` once or prefix the command
with `WANDB_MODE=offline`.

Resume from a checkpoint suffix:

```bash
torchrun --nproc_per_node=4 train.py ... --resume latest
```

## Configs

| Config | |
|---|---|
| `configs/huvr_siren.yaml` | HUVR+SIREN: `p=16`, `D=32`, 4-layer SIREN of width 256, `ω₀=10`, amplitude+shift modulation |
| `configs/huvr.yaml` | HUVR as published: `p=16`, `D=32`, 3-layer ReLU MLP, Conv+PixelShuffle upsampler |

Each ablation differs from `configs/huvr_siren.yaml` in one axis:

| | Configs | Axis |
|---|---|---|
| Patch size | `ablations/p8.yaml`, `p32.yaml` | `tokenizer.patch_size` |
| Bottleneck width | `ablations/D16.yaml`, `D128.yaml`, `D256.yaml`, `D768.yaml` | `embedding_dim` |
| SIREN base frequency | `ablations/w0_5.yaml`, `w0_30.yaml` | `hypo_siren.omega_0` |
| Modulation interface | `ablations/mod_amplitude.yaml`, `mod_shift.yaml`, `mod_incode.yaml` | `siren_modulation` |
| Training-set size | `ablations/N64.yaml`, `N256.yaml`, `N1024.yaml`, `N3338.yaml` | `terrain_subset_n`, with epochs scaled to hold the step budget fixed |

The training-set-size configs also set `terrain_augment: false`.

## Run directory

```
runs/
  checkpoints/
    terrain_<exp_name>_latest.pth      # every epoch
    terrain_<exp_name>_epoch<N>.pth    # every ckpt_freq epochs
    subset_info.json                   # subset mode only: which tiles were drawn
```

`--exp-name` defaults to the config's basename.

## Citation

```bibtex
@inproceedings{feng2026rethinking,
  title     = {Rethinking Amortized Neural Representations for High-Resolution Terrain Elevation Data},
  author    = {Feng, Haoan and Xu, Xin and De Floriani, Leila},
  booktitle = {Proceedings of the 34th ACM International Conference on Advances in Geographic Information Systems (SIGSPATIAL '26)},
  year      = {2026},
  doi       = {10.1145/3841645.3844199}
}
```

## License

MIT (see `../LICENSE`), with two upstream terms carried by the files whose
headers name them: portions derive from
[trans-inr](https://github.com/yinboc/trans-inr) (BSD 3-Clause) and from
[HUVR](https://github.com/tiktok/huvr) (MIT).
