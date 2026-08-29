"""Training entry point.

    torchrun --nproc_per_node=4 train.py \
        --method huvr_siren \
        --config configs/huvr_siren.yaml \
        --data /path/to/shards \
        --log-root runs/
"""

from __future__ import annotations

import argparse
import pathlib

import yaml

from huvr_siren.trainers import TRAINERS
from huvr_siren.utils import init_distributed_mode_torchrun, load_config


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="train.py")
    ap.add_argument("--method", required=True, choices=sorted(TRAINERS))
    ap.add_argument("--config", required=True, type=pathlib.Path)
    ap.add_argument("--data", type=str, default=None,
                    help="Shard directory; overrides terrain_data_dir in the config.")
    ap.add_argument("--log-root", type=str, default="runs",
                    help="Checkpoints, visualisations and W&B cache land here.")
    ap.add_argument("--exp-name", type=str, default=None,
                    help="Run directory name. Defaults to the config's basename.")
    ap.add_argument("--resume", type=str, default=None,
                    help="Checkpoint suffix to resume from, e.g. 'latest'.")

    # Set by torchrun; read by init_distributed_mode_torchrun.
    ap.add_argument("--nproc_per_node", default=1, type=int)
    ap.add_argument("--world_size", default=1, type=int)
    ap.add_argument("--dist_url", default="env://", type=str)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    init_distributed_mode_torchrun(args)

    cfg = load_config(
        args.config,
        args.method,
        terrain_data_dir=args.data,
        ckpt_suffix=args.resume,
    )
    cfg["exp_name"] = args.exp_name or args.config.stem
    cfg["log_root"] = args.log_root
    cfg["ngpus_per_node"] = args.nproc_per_node
    cfg["env"] = {"tot_gpus": args.world_size, "dist_url": args.dist_url}

    if args.gpu == 0:
        print("\n=== Resolved config ===")
        print(yaml.dump(cfg, default_flow_style=False, sort_keys=True))
        print("=======================\n", flush=True)

    TRAINERS[args.method](args.gpu, cfg).run()


if __name__ == "__main__":
    main()
