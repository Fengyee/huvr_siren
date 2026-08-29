"""SIREN-specific layers and initialization utilities.

Implements batched SIREN linear layer with sinusoidal activation and
proper initialization following Sitzmann et al. (2020).

Modulation follows Mehta et al. (2021): h_i = alpha_i * sin(omega_0 * (W_i h + b_i))
"""

import math

import torch


def siren_init(weight: torch.Tensor, omega_0: float, is_first: bool) -> None:
    """SIREN weight initialization (Sitzmann et al., 2020).

    First layer: Uniform(-1/fan_in, 1/fan_in)
    Subsequent layers: Uniform(-sqrt(6/fan_in)/omega_0, sqrt(6/fan_in)/omega_0)
    """
    fan_in = weight.shape[0]
    if is_first:
        bound = 1.0 / fan_in
    else:
        bound = math.sqrt(6.0 / fan_in) / omega_0
    with torch.no_grad():
        weight.uniform_(-bound, bound)


def init_siren_wb(
    shape: tuple[int, int], omega_0: float, is_first: bool
) -> torch.Tensor:
    """Initialize a packed [weight; bias] matrix of shape (D_in + 1, D_out)."""
    d_in = shape[0] - 1
    d_out = shape[1]

    weight = torch.empty(d_in, d_out)
    siren_init(weight, omega_0, is_first)

    bias = torch.empty(1, d_out)
    bound = 1.0 / d_in if is_first else math.sqrt(6.0 / d_in) / omega_0
    bias.uniform_(-bound, bound)

    return torch.cat([weight, bias], dim=0).detach()


def _batched_linear(x: torch.Tensor, wb: torch.Tensor) -> torch.Tensor:
    """Batched Wh + b, with the bias packed as the last row of `wb`.

    x: (B, N, D_in); wb: (B, D_in + 1, D_out) -> (B, N, D_out)
    """
    return torch.matmul(x, wb[:, :-1, :]) + wb[:, -1:, :]


def batched_siren_linear(
    x: torch.Tensor,
    wb: torch.Tensor,
    omega_0: float,
    modulation: torch.Tensor | None = None,
    shift: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batched SIREN linear layer with optional modulation.

    Supports amplitude modulation (alpha * sin(...)), shift modulation
    (sin(... + phi)), or both: alpha * sin(omega_0 * (Wh + b) + phi).

    Args:
        x: (B, N, D_in)
        wb: (B, D_in + 1, D_out), bias packed as the last row.
        omega_0: frequency scaling factor.
        modulation: optional (B, 1, D_out) amplitude modulation.
        shift: optional (B, 1, D_out) phase shift (added before sin).

    Returns:
        (B, N, D_out)
    """
    h = omega_0 * _batched_linear(x, wb)
    if shift is not None:
        h = h + shift
    h = torch.sin(h)
    if modulation is not None:
        h = modulation * h
    return h


def batched_siren_linear_incode(
    x: torch.Tensor,
    wb: torch.Tensor,
    omega_0: float,
    log_amplitude: torch.Tensor,
    log_frequency: torch.Tensor,
    phase: torch.Tensor,
    dc_offset: torch.Tensor,
    clamp_range: float = 5.0,
) -> torch.Tensor:
    """Batched SIREN linear layer with INCODE-style 4-signal modulation.

    Computes: exp(a) * sin(exp(b) * omega_0 * (Wh + bias) + c) + d

    Args:
        x: (B, N, D_in)
        wb: (B, D_in + 1, D_out), bias packed as the last row.
        omega_0: base frequency scaling factor.
        log_amplitude: (B, 1, D_out) — clamped then exponentiated.
        log_frequency: (B, 1, D_out) — clamped then exponentiated.
        phase: (B, 1, D_out) — additive phase shift before sin.
        dc_offset: (B, 1, D_out) — additive offset after sin.
        clamp_range: clamp log_amplitude/log_frequency to [-range, range].

    Returns:
        (B, N, D_out)
    """
    h = _batched_linear(x, wb)
    amp = torch.exp(log_amplitude.clamp(-clamp_range, clamp_range))
    freq = torch.exp(log_frequency.clamp(-clamp_range, clamp_range))
    return amp * torch.sin(freq * omega_0 * h + phase) + dc_offset


def batched_siren_linear_final(
    x: torch.Tensor, wb: torch.Tensor
) -> torch.Tensor:
    """Final SIREN layer -- linear output (no sinusoidal activation)."""
    return _batched_linear(x, wb)
