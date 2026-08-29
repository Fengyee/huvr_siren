from .huvr import HuvrTrainer
from .huvr_siren import HuvrSirenTrainer

TRAINERS = {"huvr": HuvrTrainer, "huvr_siren": HuvrSirenTrainer}

__all__ = ["HuvrTrainer", "HuvrSirenTrainer", "TRAINERS"]
