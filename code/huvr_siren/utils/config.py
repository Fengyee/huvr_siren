"""Config loading.

A config is a single YAML file. Unknown keys are rejected rather than ignored,
so a typo in a hyperparameter name fails at startup instead of silently
training something else.
"""

from __future__ import annotations

import pathlib

import yaml

# Keys every run must set, per method. No defaults: the value that ran the
# experiment belongs in the config file, not in this module.
REQUIRED = {
    "huvr": {
        "tokenizer", "hyponet", "hypocnn", "mod_idxs",
        "transformer_encoder", "transformer_decoder", "embedding_dim",
        "use_hypocnn", "use_global_token",
    },
    "huvr_siren": {
        "tokenizer", "hypo_siren",
        "transformer_encoder", "transformer_decoder", "embedding_dim",
        "input_channels", "siren_modulation", "input_proj_mode",
    },
}

REQUIRED_COMMON = {
    "batch_size", "num_workers", "lr", "clip", "max_epochs", "warmup_epochs",
    "weight_decay", "opt_eps", "ckpt_freq", "vis_freq",
    "terrain_data_dir", "terrain_num_train", "terrain_num_val",
    "terrain_augment",
}

OPTIONAL = {
    "wandb_project": "huvr-siren",
    "enable_val": True,
    "stop_epoch": None,          # defaults to max_epochs
    "pretrain_path": None,
    "ckpt_suffix": None,
    "terrain_subset_n": None,    # training-set-size sweep
    "terrain_subset_seed": None,
    "dataset_name": "terrain",
}


def load_config(path: str | pathlib.Path, method: str, **runtime) -> dict:
    """Read a YAML config for `method`, validate it, and apply runtime paths.

    `runtime` carries values that belong to an invocation rather than to an
    experiment — log_root, exp_name — and always wins over the file.
    """
    if method not in REQUIRED:
        raise ValueError(f"unknown method {method!r}; expected one of {sorted(REQUIRED)}")

    cfg = yaml.safe_load(pathlib.Path(path).read_text())
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: expected a YAML mapping")

    required = REQUIRED[method] | REQUIRED_COMMON
    known = required | set(OPTIONAL)

    unknown = sorted(set(cfg) - known)
    if unknown:
        raise ValueError(
            f"{path}: unknown config keys {unknown}. "
            f"Known keys: {sorted(known)}"
        )

    missing = sorted(required - set(cfg))
    if missing:
        raise ValueError(f"{path}: missing required keys {missing}")

    for key, default in OPTIONAL.items():
        cfg.setdefault(key, default)
    if cfg["stop_epoch"] is None:
        cfg["stop_epoch"] = cfg["max_epochs"]

    cfg.update({k: v for k, v in runtime.items() if v is not None})
    return cfg
