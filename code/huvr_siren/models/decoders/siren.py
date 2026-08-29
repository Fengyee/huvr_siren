"""Modulated SIREN implicit decoder for terrain elevation reconstruction.

Implements Modulated SIREN (Mehta et al., 2021):
    h_i = alpha_i . sin(omega_0 * (W_i h_{i-1} + b_i))

where alpha_i are per-layer modulation vectors produced by the hypernetwork
from per-patch latent codes.

Maps 2D coordinates (x, y) -> 1D elevation z.
"""

import torch
import torch.nn as nn

from .siren_layers import (
    batched_siren_linear,
    batched_siren_linear_final,
    batched_siren_linear_incode,
)


class SirenDecoder(nn.Module):

    def __init__(
        self,
        depth: int,
        in_dim: int,
        out_dim: int,
        hidden_dim: str,
        omega_0: float = 10.0,
        **compat_kwargs,
    ):
        """
        Args:
            depth: total number of weight matrices (including output layer).
            in_dim: input coordinate dimension (2 for terrain).
            out_dim: output dimension (1 for elevation).
            hidden_dim: underscore-separated string of per-layer output dims
                (excluding the output layer), e.g. '256_256_256'.
            omega_0: SIREN frequency parameter.
            **compat_kwargs: ignored, for interface compatibility.
        """
        super().__init__()
        self.depth = depth
        self.omega_0 = omega_0
        self.in_dim = in_dim
        self.out_dim = out_dim

        hidden_dims = [int(x) for x in hidden_dim.split('_')]
        hidden_dims.append(out_dim)
        assert len(hidden_dims) == depth, (
            f"depth={depth} but got {len(hidden_dims)} layer dims "
            f"from hidden_dim + out_dim"
        )

        # Base SIREN weight shapes (shared across patches, set per-batch)
        self.param_shapes: dict[str, tuple[int, int]] = {}
        # Per-layer amplitude modulation vector shapes
        self.modulation_shapes: dict[str, tuple[int]] = {}
        # Per-layer shift (phase) modulation vector shapes (same dims)
        self.shift_shapes: dict[str, tuple[int]] = {}

        last_dim = in_dim
        for i in range(depth):
            cur_dim = hidden_dims[i]
            self.param_shapes[f'wb{i}'] = (last_dim + 1, cur_dim)
            if i < depth - 1:
                self.modulation_shapes[f'mod{i}'] = (cur_dim,)
                self.shift_shapes[f'shift{i}'] = (cur_dim,)
            last_dim = cur_dim

        self.params: dict[str, torch.Tensor] | None = None
        self.modulations: dict[str, torch.Tensor] | None = None
        self.shifts: dict[str, torch.Tensor] | None = None
        self.frequencies: dict[str, torch.Tensor] | None = None
        self.dc_offsets: dict[str, torch.Tensor] | None = None
        self.use_incode: bool = False

    def set_params(self, params: dict[str, torch.Tensor]) -> None:
        self.params = params

    def set_modulations(self, modulations: dict[str, torch.Tensor]) -> None:
        self.modulations = modulations

    def set_shifts(self, shifts: dict[str, torch.Tensor]) -> None:
        self.shifts = shifts

    def set_frequencies(self, frequencies: dict[str, torch.Tensor]) -> None:
        self.frequencies = frequencies

    def set_dc_offsets(self, dc_offsets: dict[str, torch.Tensor]) -> None:
        self.dc_offsets = dc_offsets

    def set_incode_mode(self, use_incode: bool) -> None:
        self.use_incode = use_incode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, in_dim) coordinate grid.

        Returns:
            (B, H, W, out_dim) predicted elevation.
        """
        B, h, w = x.shape[0], x.shape[1], x.shape[2]
        x = x.reshape(B, -1, x.shape[-1])  # (B, H*W, in_dim)

        for i in range(self.depth):
            if i < self.depth - 1:
                if self.use_incode:
                    x = batched_siren_linear_incode(
                        x, self.params[f'wb{i}'], self.omega_0,
                        log_amplitude=self.modulations[f'mod{i}'],
                        log_frequency=self.frequencies[f'freq{i}'],
                        phase=self.shifts[f'shift{i}'],
                        dc_offset=self.dc_offsets[f'dc{i}'],
                    )
                else:
                    mod = self.modulations.get(f'mod{i}') if self.modulations else None
                    sh = self.shifts.get(f'shift{i}') if self.shifts else None
                    x = batched_siren_linear(
                        x, self.params[f'wb{i}'], self.omega_0,
                        modulation=mod, shift=sh,
                    )
            else:
                x = batched_siren_linear_final(x, self.params[f'wb{i}'])

        return x.reshape(B, h, w, -1)  # (B, H, W, out_dim)
