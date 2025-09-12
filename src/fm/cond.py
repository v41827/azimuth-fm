import math
import torch
from typing import Literal

# -----------------------------
# Azimuth feature builders
# -----------------------------

def azimuth_features_rad(az_rad: torch.Tensor, K: int = 2) -> torch.Tensor:
    """
    Build [sin(k*phi), cos(k*phi)]_{k=1..K} from radians.
    az_rad: [B] in radians.
    Returns: [B, 2*K]
    """
    phi = az_rad
    outs = []
    for k in range(1, K + 1):
        outs.append(torch.sin(k * phi))
        outs.append(torch.cos(k * phi))
    return torch.stack(outs, dim=-1)

def azimuth_features_deg(az_deg: torch.Tensor, K: int = 2) -> torch.Tensor:
    """
    Build [sin(k*phi), cos(k*phi)]_{k=1..K} from degrees.
    az_deg: [B] in degrees.
    Returns: [B, 2*K]
    """
    phi = az_deg * math.pi / 180.0
    outs = []
    for k in range(1, K + 1):
        outs.append(torch.sin(k * phi))
        outs.append(torch.cos(k * phi))
    return torch.stack(outs, dim=-1)


def azimuth_features_sincos(az_sin: torch.Tensor, az_cos: torch.Tensor, K: int = 2) -> torch.Tensor:
    """
    Same as above but start from provided sin/cos of the fundamental angle.
    We recover phi with atan2(sin, cos), then build higher harmonics.
    Shapes: az_sin, az_cos: [B]
    Returns: [B, 2*K]
    """
    phi = torch.atan2(az_sin, az_cos)
    outs = []
    for k in range(1, K + 1):
        outs.append(torch.sin(k * phi))
        outs.append(torch.cos(k * phi))
    return torch.stack(outs, dim=-1)


# -----------------------------
# Time embedding
# -----------------------------

def t_embed(t: torch.Tensor, dim: int = 64) -> torch.Tensor:
    """Sin/cos positional embedding for continuous t in [0,1].
    Returns [B, dim].
    """
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(math.log(1.0), math.log(1000.0), half, device=t.device)
    )
    args = t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# -----------------------------
# Combiners (build final FiLM condition)
# -----------------------------

def build_cond(
    t_scalar: torch.Tensor,
    *,
    az_source: Literal["deg", "rad", "sincos"] = "deg",
    az_deg: torch.Tensor | None = None,
    az_rad: torch.Tensor | None = None,
    az_sin: torch.Tensor | None = None,
    az_cos: torch.Tensor | None = None,
    fourier_feats: int = 2,
    t_dim: int = 64,
) -> torch.Tensor:
    """
    Build conditioning vector [ azimuth Fourier features || t-embed ].

    Usage patterns:
      - build_cond(t, az_source="deg",   az_deg=az_deg)
      - build_cond(t, az_source="rad",   az_rad=az_rad)
      - build_cond(t, az_source="sincos", az_sin=az_sin, az_cos=az_cos)

    Returns: [B, 2*fourier_feats + t_dim]
    """
    device = t_scalar.device
    if az_source == "deg":
        assert az_deg is not None, "az_deg must be provided when az_source='deg'"
        azf = azimuth_features_deg(az_deg.to(device), K=fourier_feats)
    elif az_source == "rad":
        assert az_rad is not None, "az_rad must be provided when az_source='rad'"
        azf = azimuth_features_rad(az_rad.to(device), K=fourier_feats)
    elif az_source == "sincos":
        assert az_sin is not None and az_cos is not None, (
            "az_sin and az_cos must be provided when az_source='sincos'"
        )
        azf = azimuth_features_sincos(az_sin.to(device), az_cos.to(device), K=fourier_feats)
    else:
        raise ValueError(f"Unknown az_source: {az_source}")

    tf = t_embed(t_scalar.to(device), dim=t_dim)
    return torch.cat([azf, tf], dim=-1)


# Backwards-compatible helpers (used by older training code)

def build_cond_from_deg(az_deg: torch.Tensor, t_scalar: torch.Tensor, fourier_feats: int = 2, t_dim: int = 64) -> torch.Tensor:
    return build_cond(t_scalar, az_source="deg", az_deg=az_deg, fourier_feats=fourier_feats, t_dim=t_dim)


def build_cond_from_sincos(az_sin: torch.Tensor, az_cos: torch.Tensor, t_scalar: torch.Tensor, fourier_feats: int = 2, t_dim: int = 64) -> torch.Tensor:
    return build_cond(t_scalar, az_source="sincos", az_sin=az_sin, az_cos=az_cos, fourier_feats=fourier_feats, t_dim=t_dim)


def build_cond_from_rad(az_rad: torch.Tensor, t_scalar: torch.Tensor, fourier_feats: int = 2, t_dim: int = 64) -> torch.Tensor:
    return build_cond(t_scalar, az_source="rad", az_rad=az_rad, fourier_feats=fourier_feats, t_dim=t_dim)