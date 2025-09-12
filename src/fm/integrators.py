from __future__ import annotations
import torch
from typing import Tuple

from .cond import build_cond


@torch.no_grad()
def heun_integrate(
    model,
    shape: Tuple[int, int, int],
    az: torch.Tensor,
    n_steps: int,
    device: torch.device,
    *,
    fourier_feats: int = 2,
    t_dim: int = 64,
    az_source: str | None = None,
):
    """
    Heun's method ODE solver for Flow Matching.

    Integrates dx/dt = v_theta(x, t | az) from t=0 → 1.

    az can be degrees [B] or sin/cos [B,2]. If az_source is None, it is auto-detected.
    """
    B = az.shape[0]
    C, F, T = shape
    x = torch.randn((B, C, F, T), device=device)

    if az_source is None:
        az_source = "sincos" if (az.ndim == 2 and az.shape[1] == 2) else "deg"

    t_grid = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
    for i in range(n_steps):
        t0, t1 = t_grid[i], t_grid[i + 1]
        dt = t1 - t0
        t0b = torch.full((B,), float(t0), device=device)
        t1b = torch.full((B,), float(t1), device=device)

        if az_source == "deg":
            h0 = build_cond(t0b, az_source="deg", az_deg=az, fourier_feats=fourier_feats, t_dim=t_dim)
            v0 = model(x, h0)
            x_pred = x + dt * v0
            h1 = build_cond(t1b, az_source="deg", az_deg=az, fourier_feats=fourier_feats, t_dim=t_dim)
            v1 = model(x_pred, h1)
        else:
            h0 = build_cond(
                t0b, az_source="sincos", az_sin=az[:, 0], az_cos=az[:, 1],
                fourier_feats=fourier_feats, t_dim=t_dim,
            )
            v0 = model(x, h0)
            x_pred = x + dt * v0
            h1 = build_cond(
                t1b, az_source="sincos", az_sin=az[:, 0], az_cos=az[:, 1],
                fourier_feats=fourier_feats, t_dim=t_dim,
            )
            v1 = model(x_pred, h1)

        x = x + dt * 0.5 * (v0 + v1)

    return x
