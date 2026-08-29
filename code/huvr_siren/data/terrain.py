"""WebDataset loader for terrain shards.

A shard is a `.tar` named `shard-{:06d}.tar` holding two files per sample:

    {key}.elevation.npy    raw elevation in metres, float32, (256, 256)
    {key}.metadata.json    optional; only `tile_id` is read, to caption figures

Elevation is stored unnormalised: the trainer min-max normalises each tile
from the tile itself, so a shard carries no statistics of its own.

    from huvr_siren.data.terrain import create_terrain_dataloader

    loader = create_terrain_dataloader("shards/train", batch_size=8)
    for batch in loader:
        batch["elevation"]   # (B, 1, 256, 256) float32, metres
"""

import glob
import io
import os
import json
from typing import Any

import numpy as np
import torch
import webdataset as wds


def check_tile_batch(elevation: torch.Tensor, input_size: int) -> None:
    """Reject tiles the patch grid cannot describe.

    The number of patches is derived from `tokenizer.input_size`, not from the
    tensor, so a tile of another size or a non-square one stitches back
    together wrong instead of failing.
    """
    _, c, h, w = elevation.shape
    if (c, h, w) != (1, input_size, input_size):
        raise ValueError(
            f"expected tiles of shape (B, 1, {input_size}, {input_size}) to match "
            f"tokenizer.input_size={input_size}, got (B, {c}, {h}, {w}). "
            "Crop or resample to a square tile of that size first."
        )


def decode_elevation(value: bytes) -> torch.Tensor:
    """Decode .npy bytes to (1, H, W) float32 tensor."""
    buf = io.BytesIO(value)
    arr = np.load(buf)  # (H, W) float32
    return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)


def decode_metadata(value: bytes) -> dict[str, Any]:
    """Decode metadata JSON bytes to dict."""
    return json.loads(value.decode("utf-8"))


def terrain_decoder(sample: dict) -> dict:
    """Decode a WebDataset sample into tensors and metadata."""
    raw = sample.get("metadata.json")
    return {
        "elevation": decode_elevation(sample["elevation.npy"]),
        "metadata": decode_metadata(raw) if raw is not None else {},
    }


def augment_terrain(sample: dict) -> dict:
    """Apply random geometric augmentations to a terrain sample.

    Augmentations (all preserve fixed pixel resolution):
      - Random horizontal flip
      - Random vertical flip
      - Random 90-degree rotation (0, 90, 180, 270)
    """
    elev = sample["elevation"]
    if np.random.rand() < 0.5:
        elev = elev.flip(-1)
    if np.random.rand() < 0.5:
        elev = elev.flip(-2)
    k = np.random.randint(0, 4)
    if k > 0:
        elev = torch.rot90(elev, k, dims=(-2, -1))
    sample["elevation"] = elev
    return sample


def collate_terrain(samples: list[dict]) -> dict:
    """Collate terrain samples into a batch.

    ``image_id`` is included only if all samples carry it (i.e., when produced by
    ``TerrainSubsetDataset``). For the streaming WebDataset path it is absent.
    """
    out = {
        "elevation": torch.stack([s["elevation"] for s in samples]),
        "metadata": [s["metadata"] for s in samples],
    }
    if all("image_id" in s for s in samples):
        out["image_id"] = torch.tensor(
            [s["image_id"] for s in samples], dtype=torch.long,
        )
    return out


class TerrainSubsetDataset(torch.utils.data.Dataset):
    """In-memory dataset holding a fixed subset of terrain samples.

    Each sample carries a deterministic integer ``image_id`` equal to its
    position in the subset list, stable across epochs.
    """

    def __init__(self, samples: list[dict]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = dict(self.samples[idx])  # shallow copy so we don't mutate cache
        sample["image_id"] = idx
        return sample


def create_terrain_subset_dataloader(
    data_dir: str,
    n: int = 4,
    batch_size: int | None = None,
    seed: int | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[torch.utils.data.DataLoader, list[dict]]:
    """Load a fixed subset of *n* samples from the WebDataset shards.

    When ``seed`` is None, selection is deterministic and takes the
    first *n* samples (legacy behavior — preserves the overfit
    experiment's tile identities).

    When ``seed`` is an integer, we read ALL samples, deterministically
    shuffle with ``random.Random(seed)``, and take the first *n*. This
    yields nested subsets: with the same seed, a smaller *n* is a
    strict subset of a larger *n*, enabling a clean scaling-curve
    experiment.

    When ``world_size > 1``, the selected *n* samples are split across
    ranks via ``samples[rank::world_size]`` so that DDP training sees
    each sample exactly once per epoch across all GPUs. The returned
    metadata is identical across ranks (full subset identity), but each
    rank's DataLoader only yields its slice.

    Args:
        data_dir: Path to shard directory (e.g., "data/terrain_wds/train").
        n: Number of samples to extract.
        batch_size: If None, defaults to n (one batch = all samples).
        seed: If set, enables seeded nested selection.
        rank: DDP rank of this process.
        world_size: Total DDP ranks.

    Returns:
        (dataloader, metadata_list) — metadata_list records the full
        subset identity (all n samples) regardless of rank.
    """
    import random

    shard_paths = sorted(glob.glob(f"{data_dir}/shard-*.tar"))
    if not shard_paths:
        raise FileNotFoundError(f"No shards found in {data_dir}")

    # Use a passthrough nodesplitter: every rank reads ALL shards and
    # constructs the same full sample list. We then apply the seeded
    # shuffle (same on all ranks) and slice per-rank below. This avoids
    # WebDataset's `single_node_only` check, which errors under DDP
    # even when we only want to materialize samples into memory.
    def _passthrough_nodesplitter(src):
        yield from src

    dataset = wds.WebDataset(
        shard_paths,
        shardshuffle=False,
        nodesplitter=_passthrough_nodesplitter,
        empty_check=False,
    ).map(terrain_decoder)

    samples: list[dict] = []
    if seed is None:
        for sample in dataset:
            samples.append(sample)
            if len(samples) >= n:
                break
        if len(samples) < n:
            raise ValueError(
                f"Requested {n} samples but only found {len(samples)}"
            )
    else:
        samples = list(dataset)
        if len(samples) < n:
            raise ValueError(
                f"Requested {n} samples but only found {len(samples)}"
            )
        random.Random(seed).shuffle(samples)
        samples = samples[:n]

    # Metadata reflects the full subset identity (used for logging).
    metadata_list = [s["metadata"] for s in samples]

    # Distribute across DDP ranks. Each rank keeps its slice only.
    if world_size > 1:
        samples = samples[rank::world_size]

    if batch_size is None:
        batch_size = max(1, len(samples))

    # Ensure batch size doesn't exceed per-rank samples.
    effective_batch_size = min(batch_size, max(1, len(samples)))

    loader = torch.utils.data.DataLoader(
        TerrainSubsetDataset(samples),
        batch_size=effective_batch_size,
        shuffle=False,
        collate_fn=collate_terrain,
        num_workers=0,
        pin_memory=True,
    )
    return loader, metadata_list


def create_terrain_dataloader(
    data_dir: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    buffer_size: int = 1000,
    distributed: bool = False,
    augment: bool = False,
    replicate: bool = False,
) -> wds.WebLoader:
    """Create a WebDataset-based DataLoader for terrain patches.

    Args:
        data_dir: Path to shard directory (e.g., "data/terrain_wds/train").
        batch_size: Batch size.
        shuffle: Whether to shuffle (uses a shuffle buffer).
        num_workers: Number of data loading workers.
        buffer_size: Size of the shuffle buffer (only used if shuffle=True).
        distributed: If True, use ``split_by_node`` so each rank reads a
            disjoint set of shards.
        augment: If True, apply random flips and 90-deg rotations.
        replicate: If True, use a passthrough nodesplitter so every rank
            reads ALL shards independently. Use for val sets that need
            to be evaluated identically on every rank without cross-rank
            communication. Mutually exclusive with ``distributed=True``.

    Returns:
        A WebLoader yielding batches with keys "elevation" and "metadata".
    """
    shard_paths = sorted(glob.glob(f"{data_dir}/shard-*.tar"))
    if not shard_paths:
        raise FileNotFoundError(f"No shards found in {data_dir}")

    if distributed and replicate:
        raise ValueError("distributed and replicate are mutually exclusive")

    if distributed:
        # split_by_node hands each rank urls[rank::world_size]. With fewer
        # shards than ranks, the starved ranks yield nothing and the job hangs
        # in the first collective instead of failing.
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if len(shard_paths) < world_size:
            raise ValueError(
                f"{data_dir} has {len(shard_paths)} shard(s) but "
                f"WORLD_SIZE={world_size}; shards are split across ranks, so "
                "pack the split into at least one shard per rank."
            )

    if replicate:
        def _passthrough(src):
            yield from src
        nodesplitter = _passthrough
    elif distributed:
        nodesplitter = wds.shardlists.split_by_node
    else:
        nodesplitter = wds.shardlists.single_node_only

    dataset = wds.WebDataset(
        shard_paths,
        shardshuffle=shuffle,
        nodesplitter=nodesplitter,
        empty_check=False,
    ).map(terrain_decoder)

    if augment:
        dataset = dataset.map(augment_terrain)

    if shuffle:
        dataset = dataset.shuffle(buffer_size)

    loader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_terrain,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader
