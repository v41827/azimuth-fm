import os, math, argparse, json, random
from datetime import datetime
from typing import List, Dict

import torch
import numpy as np

from src.models.unet_film import TinyUNet
from src.fm.integrators import heun_integrate
from src.data.dataset import BinauralSTFTDataset
from src.utils.audio import istft_wave
import torchaudio


def circular_error_deg(a: float, b: float) -> float:
    d = abs((a - b) % 360.0)
    return d if d <= 180.0 else 360.0 - d

def signed_circular_diff(a: float, b: float) -> float:
    # signed minimal difference in degrees in [-180,180)
    d = (a - b + 180.0) % 360.0 - 180.0
    return d


@torch.no_grad()
def gcc_phat_tdoa(l: torch.Tensor, r: torch.Tensor, max_lag: int) -> int:
    N = l.numel()
    nfft = 1
    while nfft < 2 * N:
        nfft *= 2
    L = torch.fft.rfft(l, n=nfft)
    R = torch.fft.rfft(r, n=nfft)
    X = L * torch.conj(R)
    X = X / torch.clamp(torch.abs(X), min=1e-8)
    xcorr = torch.fft.irfft(X, n=nfft)
    xcorr = torch.roll(xcorr, shifts=N // 2, dims=0)
    mid = nfft // 2
    low = max(0, mid - max_lag)
    high = min(nfft, mid + max_lag + 1)
    window = xcorr[low:high]
    idx = torch.argmax(window).item()
    return int(idx + low - mid)


def tdoa_to_lateral_deg(tau: float, ear_dist_m: float = 0.18, c: float = 343.0) -> float:
    s = (tau * c) / ear_dist_m
    s = max(-1.0, min(1.0, s))
    return math.degrees(math.asin(s))


def lateral_to_sofa_az(lat: float) -> float:
    return lat if lat >= 0 else (360.0 + lat)


def seen_grid() -> List[float]:
    return list(np.arange(0, 91, 5, dtype=float)) + list(np.arange(270, 360, 5, dtype=float))


def unseen_midpoints() -> List[float]:
    mids = list(np.arange(2.5, 90, 5, dtype=float)) + list(np.arange(272.5, 355, 5, dtype=float))
    mids.append(357.5)  # wrap-around between 355 and 0
    return mids


def random_front(N: int, seed: int = 0) -> List[float]:
    rng = random.Random(seed)
    out = []
    for i in range(N):
        if i % 2 == 0:
            out.append(rng.uniform(0.0, 90.0))
        else:
            out.append(rng.uniform(270.0, 360.0))
    return out


def main():
    ap = argparse.ArgumentParser(description="Benchmark GCC‑PHAT relocalization: seen, midpoints, random")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--n_per", type=int, default=20, help="samples per angle")
    ap.add_argument("--dense_n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ear_dist_m", type=float, default=0.18)
    ap.add_argument("--max_lag_ms", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save_audio", action="store_true", help="Save generated WAVs for each sample")
    ap.add_argument("--wandb_project", default=None, help="If set, log results to W&B under this project")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load checkpoint and config
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("cfg", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    stft_cfg = cfg.get("stft", {})
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

    # STFT shape
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
        C = 4; F = n_fft // 2 + 1; T = int(np.ceil((sr * seg_sec + n_fft) / hop))
    shape = (C, F, T)

    # Output
    out_dir = args.out or os.path.join("runs", "relocal_bench", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    sets: Dict[str, List[float]] = {
        "seen": seen_grid(),
        "mid": unseen_midpoints(),
        "rand": random_front(args.dense_n, args.seed),
    }

    max_lag = int((args.max_lag_ms / 1000.0) * sr)
    length = int(sr * seg_sec)

    # optional W&B
    wb = None
    if args.wandb_project is not None:
        try:
            import wandb
            wb = wandb
            wb.init(project=args.wandb_project, config={"steps": args.steps, "n_per": args.n_per, "dense_n": args.dense_n})
        except Exception:
            wb = None

    summary = {}
    for name, degs in sets.items():
        rows = []
        for deg in degs:
            az = torch.tensor([float(deg)], device=device)
            errs = []
            rows_deg = []
            for _ in range(args.n_per):
                Xgen = heun_integrate(
                    model, shape, az, int(args.steps), device,
                    fourier_feats=fourier_feats, t_dim=t_dim, az_source="deg",
                )
                L = istft_wave(Xgen[0,0:2], n_fft, hop, window, length, normalized=normalized, win_length=win_length)
                R = istft_wave(Xgen[0,2:4], n_fft, hop, window, length, normalized=normalized, win_length=win_length)
                l = L - L.mean(); r = R - R.mean()
                lag = gcc_phat_tdoa(l, r, max_lag)
                tau = lag / float(sr)
                lat = tdoa_to_lateral_deg(tau, ear_dist_m=args.ear_dist_m)
                az_est = lateral_to_sofa_az(lat) % 360.0
                err = circular_error_deg(az_est, deg)
                row = {"deg": float(deg), "tau_samples": int(lag), "tau_ms": tau*1000.0, "lat_deg": float(lat), "az_est_deg": float(az_est), "abs_err_deg": float(err)}
                rows.append(row)
                rows_deg.append(row)
                errs.append(err)
                # Save audio if requested
                if args.save_audio:
                    outa = os.path.join(out_dir, "audio", name, f"deg{int(round(deg)):03d}")
                    os.makedirs(outa, exist_ok=True)
                    wav = torch.stack([L, R], dim=0).cpu()
                    fn = os.path.join(outa, f"s{args.steps}_est{az_est:.1f}_err{err:.1f}.wav")
                    torchaudio.save(fn, wav, sr)
            # Log per-angle stats to W&B
            if wb is not None and rows_deg:
                table = wb.Table(data=[
                    [r["deg"], r["az_est_deg"], r["abs_err_deg"], r["tau_ms"]] for r in rows_deg
                ], columns=["deg", "az_est", "abs_err", "tau_ms"])
                wb.log({f"{name}/angle_{int(round(deg))}": table})
        # Save rows
        import csv
        csv_path = os.path.join(out_dir, f"{name}.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

        # Micro / Macro MAE and dispersion
        # micro: mean over all rows
        abs_all = np.array([r["abs_err_deg"] for r in rows], dtype=float)
        micro = float(np.mean(abs_all)) if abs_all.size else float("nan")
        # macro: per-angle MAE then average
        per_angle = {}
        for r in rows:
            per_angle.setdefault(r["deg"], []).append(r["abs_err_deg"])
        macro = float(np.mean([np.mean(v) for v in per_angle.values()])) if per_angle else float("nan")

        # Standard deviation (ordinary) over absolute error
        std_abs = float(np.std(abs_all)) if abs_all.size else float("nan")

        # Circular std over signed angle errors (pred - gt, wrapped)
        signed_errs = np.array([signed_circular_diff(r["az_est_deg"], r["deg"]) for r in rows], dtype=float)
        if signed_errs.size:
            rad = np.deg2rad(signed_errs)
            C = np.mean(np.cos(rad)); S = np.mean(np.sin(rad))
            R = np.sqrt(C*C + S*S)
            circ_std_deg = float(np.rad2deg(np.sqrt(max(0.0, -2.0*np.log(max(1e-12, R))))))
        else:
            circ_std_deg = float("nan")

        summary[name] = {"micro_mae": micro, "macro_mae": macro, "std_abs": std_abs, "circ_std_deg": circ_std_deg, "n_rows": len(rows), "n_angles": len(per_angle)}
        print(f"[{name}] micro-MAE={micro:.2f}°, macro-MAE={macro:.2f}°, std={std_abs:.2f}°, circ-std={circ_std_deg:.2f}° | angles={len(per_angle)} rows={len(rows)} → {csv_path}")

        if wb is not None:
            wb.log({
                f"{name}/micro_mae": micro,
                f"{name}/macro_mae": macro,
                f"{name}/std_abs": std_abs,
                f"{name}/circ_std": circ_std_deg,
            })

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {os.path.join(out_dir, 'summary.json')}")
    if wb is not None:
        try:
            import wandb
            art = wandb.Artifact("azfm-relocal-bench", type="relocal")
            for name in sets.keys():
                p = os.path.join(out_dir, f"{name}.csv")
                if os.path.isfile(p): art.add_file(p)
            art.add_file(os.path.join(out_dir, "summary.json"))
            wandb.log_artifact(art)
        except Exception:
            pass


if __name__ == "__main__":
    main()
