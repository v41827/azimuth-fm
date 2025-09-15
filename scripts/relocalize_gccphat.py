import os, math, argparse, json
from datetime import datetime
from typing import List

import torch
import numpy as np

from src.models.unet_film import TinyUNet
from src.fm.integrators import heun_integrate
from src.data.dataset import BinauralSTFTDataset
from src.utils.audio import istft_wave


def parse_degs(s: str) -> List[float]:
    # accepts formats like "0,30,48" or "[0,30,48]"
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    if not s:
        return []
    return [float(x) for x in s.split(",")]


def circular_error_deg(a: float, b: float) -> float:
    d = abs((a - b) % 360.0)
    return d if d <= 180.0 else 360.0 - d


@torch.no_grad()
def gcc_phat_tdoa(l: torch.Tensor, r: torch.Tensor, max_lag: int) -> int:
    # l, r: [N] mono waves (float32), zero-mean
    N = l.numel()
    nfft = 1
    while nfft < 2 * N:
        nfft *= 2
    L = torch.fft.rfft(l, n=nfft)
    R = torch.fft.rfft(r, n=nfft)
    X = L * torch.conj(R)
    denom = torch.clamp(torch.abs(X), min=1e-8)
    X = X / denom
    xcorr = torch.fft.irfft(X, n=nfft)
    # put zero lag in the center
    xcorr = torch.roll(xcorr, shifts=N // 2, dims=0)
    mid = nfft // 2
    low = max(0, mid - max_lag)
    high = min(nfft, mid + max_lag + 1)
    window = xcorr[low:high]
    idx = torch.argmax(window).item()
    lag = idx + low - mid
    return int(lag)


def tdoa_to_lateral_deg(tau: float, sr: int, ear_dist_m: float = 0.18, c: float = 343.0) -> float:
    # tau seconds → lateral angle in degrees (−90..+90), Woodworth model
    s = (tau * c) / ear_dist_m
    s = max(-1.0, min(1.0, s))
    return math.degrees(math.asin(s))


def lateral_to_sofa_az(lat: float) -> float:
    # map lateral [−90,90] to SOFA azimuth [0,360): left positive
    return lat if lat >= 0 else (360.0 + lat)


def main():
    ap = argparse.ArgumentParser(description="Standalone GCC‑PHAT relocalization on generated audio")
    ap.add_argument("--ckpt", required=True, help="Path to trained checkpoint (ckpt_*.pt)")
    ap.add_argument("--degs", required=True, help="Comma list or [list] of SOFA azimuths, e.g. '[0,30,48,330]'")
    ap.add_argument("--n_per_deg", type=int, default=4, help="How many samples per angle")
    ap.add_argument("--steps", type=int, default=40, help="Heun sampler steps")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ear_dist_m", type=float, default=0.18, help="Interaural distance (m)")
    ap.add_argument("--max_lag_ms", type=float, default=1.0, help="GCC‑PHAT search window (ms)")
    ap.add_argument("--out", default=None, help="Output dir (default runs/relocal/<timestamp>)")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load checkpoint and restore model + config
    ckpt = torch.load(args.ckpt, map_location="cpu")
    saved = ckpt.get("cfg", {})
    # pull hyperparams
    model_cfg = saved.get("model", {})
    data_cfg = saved.get("data", {})
    stft_cfg = saved.get("stft", {})
    sr = int(data_cfg.get("sr", 16000))
    seg_sec = float(data_cfg.get("seg_sec", 1.5))
    n_fft = int(stft_cfg.get("n_fft", 512))
    hop = int(stft_cfg.get("hop", 256))
    window = stft_cfg.get("window", "hann")
    normalized = bool(stft_cfg.get("normalized", True))
    win_length = int(stft_cfg.get("win_length", n_fft))
    fourier_feats = int(model_cfg.get("fourier_feats", 2))
    t_dim = int(model_cfg.get("t_dim", 64))

    cond_dim = 2 * fourier_feats + t_dim
    model = TinyUNet(base=int(model_cfg.get("base_ch", 96)), depth=int(model_cfg.get("depth", 4)), cond_dim=cond_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Determine STFT shape: use a one‑item dataset for shape only (no content used)
    try:
        ds = BinauralSTFTDataset(
            data_cfg.get("val_csv", data_cfg.get("train_csv")),
            data_cfg.get("audio_root", "."), sr=sr, seg_sec=seg_sec,
            n_fft=n_fft, hop=hop, win=window, normalized=normalized, win_length=win_length,
            az_mode=data_cfg.get("az_mode", "sincos"), enforce_csv_sr=False,
        )
        X0, _, _ = ds[0]
        C, F, T = X0.shape[0], X0.shape[1], X0.shape[2]
    except Exception:
        # Fallback: approximate F,T (works with center=True default in torch.stft)
        C = 4
        N = int(sr * seg_sec)
        # Conservative T estimate
        T = int(np.ceil((N + n_fft) / hop))
        F = n_fft // 2 + 1
    shape = (C, F, T)

    # Output dir
    out_dir = args.out or os.path.join("runs", "relocal", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    degs = parse_degs(args.degs)
    max_lag = int((args.max_lag_ms / 1000.0) * sr)

    rows = []
    for deg in degs:
        az = torch.tensor([float(deg)], device=device)
        for j in range(args.n_per_deg):
            Xgen = heun_integrate(
                model, shape, az, int(args.steps), device,
                fourier_feats=fourier_feats, t_dim=t_dim, az_source="deg",
            )
            length = int(sr * seg_sec)
            L = istft_wave(Xgen[0,0:2], n_fft, hop, window, length, normalized=normalized, win_length=win_length)
            R = istft_wave(Xgen[0,2:4], n_fft, hop, window, length, normalized=normalized, win_length=win_length)
            l = L - L.mean(); r = R - R.mean()
            lag = gcc_phat_tdoa(l, r, max_lag)
            tau = lag / float(sr)
            lat = tdoa_to_lateral_deg(tau, sr, ear_dist_m=args.ear_dist_m)
            az_est = lateral_to_sofa_az(lat) % 360.0
            err = circular_error_deg(az_est, deg)
            rows.append({
                "prompt_deg": float(deg),
                "tau_samples": int(lag),
                "tau_ms": tau * 1000.0,
                "lat_deg": float(lat),
                "az_est_deg": float(az_est),
                "abs_err_deg": float(err),
            })

    # Save and print summary
    out_json = os.path.join(out_dir, "results.json")
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)

    mae = float(np.mean([r["abs_err_deg"] for r in rows])) if rows else float("nan")
    print("Relocalization results (GCC‑PHAT)")
    print(f" - N samples: {len(rows)}  |  MAE: {mae:.2f} deg")
    print(" - First 10 rows:")
    for r in rows[:10]:
        print(f"   prompt={r['prompt_deg']:.1f}°, tau={r['tau_ms']:.3f} ms, lat={r['lat_deg']:.1f}°, az_est={r['az_est_deg']:.1f}°, abs_err={r['abs_err_deg']:.1f}°")
    print(f"Saved full results to {out_json}\nOutput dir: {out_dir}")


if __name__ == "__main__":
    main()

