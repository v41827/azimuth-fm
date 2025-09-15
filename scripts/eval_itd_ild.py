import os, math, csv
from typing import List, Dict, Tuple

import torch
from torch.utils.data import DataLoader
from hydra import main
from omegaconf import DictConfig, OmegaConf

from src.data.dataset import BinauralSTFTDataset
from src.models.unet_film import TinyUNet
from src.fm.integrators import heun_integrate
from src.utils.audio import istft_wave
from src.utils.metrics import itd_ild_proxy


def lateral_deg_from_global(deg: float) -> float:
    # Map global azimuth [0,360) to signed lateral angle in [-90, 90]
    d = deg % 360.0
    if d > 180.0:
        d = d - 360.0  # now in (-180, 180]
    # fold front/back, keep left negative, right positive
    if d > 90.0:
        d = 180.0 - d
    if d < -90.0:
        d = -180.0 - d
    return d


def gcc_phat_tdoa(l: torch.Tensor, r: torch.Tensor, max_lag: int) -> int:
    # l, r: [N] on same device, zero-mean
    N = l.shape[0]
    nfft = 1
    while nfft < 2 * N:
        nfft *= 2
    L = torch.fft.rfft(l, n=nfft)
    R = torch.fft.rfft(r, n=nfft)
    X = L * torch.conj(R)
    denom = torch.clamp(torch.abs(X), min=1e-8)
    X /= denom
    xcorr = torch.fft.irfft(X, n=nfft)
    xcorr = torch.roll(xcorr, shifts=N//2, dims=0)  # center lag 0
    mid = nfft // 2
    low = max(0, mid - max_lag)
    high = min(nfft, mid + max_lag + 1)
    window = xcorr[low:high]
    idx = torch.argmax(window).item()
    lag = idx + low - mid
    return int(lag)


def tdoa_to_lateral_deg(lag: int, sr: int, ear_dist_m: float = 0.18, c: float = 343.0) -> float:
    # Woodworth model for spherical head: tau = (d/c) * sin(theta)
    tau = lag / float(sr)
    s = (tau * c) / ear_dist_m
    s = float(max(-1.0, min(1.0, s)))
    theta = math.degrees(math.asin(s))
    return theta


def circular_diff_deg(a: float, b: float) -> float:
    d = abs((a - b) % 360.0)
    return d if d <= 180.0 else 360.0 - d


@torch.no_grad()
def compute_templates(cfg: DictConfig, device: torch.device, batch_size: int) -> Dict[int, Dict[str, float]]:
    ds = BinauralSTFTDataset(
        cfg.data.val_csv, cfg.data.audio_root, sr=cfg.data.sr, seg_sec=cfg.data.seg_sec,
        n_fft=cfg.stft.n_fft, hop=cfg.stft.hop, win=cfg.stft.window,
        normalized=cfg.stft.normalized, win_length=cfg.stft.win_length,
        az_mode=cfg.data.az_mode, enforce_csv_sr=True,
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=cfg.train.num_workers if hasattr(cfg, 'train') else 2)
    buckets: Dict[int, Dict[str, List[float]]] = {}
    for X, az, N in dl:
        X = X.to(device)
        az = torch.as_tensor(az, device=device, dtype=torch.float32)
        itd, ild = itd_ild_proxy(X)
        B = X.shape[0]
        for i in range(B):
            # reconstruct to compute GCC/IACC
            L = istft_wave(X[i, 0:2], cfg.stft.n_fft, cfg.stft.hop, cfg.stft.window, int(N[i]),
                           normalized=cfg.stft.normalized, win_length=cfg.stft.win_length)
            R = istft_wave(X[i, 2:4], cfg.stft.n_fft, cfg.stft.hop, cfg.stft.window, int(N[i]),
                           normalized=cfg.stft.normalized, win_length=cfg.stft.win_length)
            l = L - L.mean(); r = R - R.mean()
            max_lag = int((cfg.eval.max_lag_ms / 1000.0) * cfg.data.sr)
            lag = gcc_phat_tdoa(l, r, max_lag)
            lat = tdoa_to_lateral_deg(lag, cfg.data.sr)
            # simple IACC: max of normalized xcorr in lag window
            n = torch.norm(l) * torch.norm(r) + 1e-8
            # reuse correlation window from gcc
            Nsig = l.numel()
            nfft = 1
            while nfft < 2 * Nsig:
                nfft *= 2
            Lf = torch.fft.rfft(l, n=nfft)
            Rf = torch.fft.rfft(r, n=nfft)
            xcorr = torch.fft.irfft(Lf * torch.conj(Rf), n=nfft)
            xcorr = torch.roll(xcorr, shifts=Nsig//2, dims=0)
            mid = nfft // 2
            low = max(0, mid - max_lag); high = min(nfft, mid + max_lag + 1)
            iacc = float((xcorr[low:high].abs().max() / n).item())

            if cfg.data.az_mode == 'sincos' and az.ndim == 2:
                deg = math.degrees(math.atan2(float(az[i,0]), float(az[i,1])))
            else:
                deg = float(az[i].item())
            deg = int(round(deg)) % 360
            buckets.setdefault(deg, {"itd": [], "ild": [], "lat": [], "iacc": []})
            buckets[deg]["itd"].append(float(itd[i].item()))
            buckets[deg]["ild"].append(float(ild[i].item()))
            buckets[deg]["lat"].append(float(lat))
            buckets[deg]["iacc"].append(float(iacc))

    # reduce to means
    out: Dict[int, Dict[str, float]] = {}
    for k, v in buckets.items():
        out[k] = {m: float(torch.tensor(v[m]).mean().item()) for m in v}
    return out


@main(config_path="../configs", config_name="config", version_base="1.3")
def run(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    os.makedirs(cfg.eval.out_dir, exist_ok=True)
    device = torch.device(cfg.eval.device if torch.cuda.is_available() else 'cpu')

    # Reference templates from validation data
    templates = compute_templates(cfg, device, cfg.eval.batch_size)

    # Build model with cond_dim from cfg
    cond_dim = 2 * cfg.model.fourier_feats + cfg.model.t_dim
    model = TinyUNet(base=cfg.model.base_ch, depth=cfg.model.depth, cond_dim=cond_dim).to(device)
    ckpt = torch.load(cfg.eval.ckpt, map_location='cpu')
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Determine STFT shape from one val batch
    ds = BinauralSTFTDataset(
        cfg.data.val_csv, cfg.data.audio_root, sr=cfg.data.sr, seg_sec=cfg.data.seg_sec,
        n_fft=cfg.stft.n_fft, hop=cfg.stft.hop, win=cfg.stft.window,
        normalized=cfg.stft.normalized, win_length=cfg.stft.win_length,
        az_mode=cfg.data.az_mode, enforce_csv_sr=True,
    )
    X0, _, _ = next(iter(DataLoader(ds, batch_size=1)))
    C, F, T = X0.shape[1:]
    shape = (C, F, T)

    results: List[Dict[str, float]] = []
    max_lag = int((cfg.eval.max_lag_ms / 1000.0) * cfg.data.sr)
    length = int(cfg.data.seg_sec * cfg.data.sr)

    for deg in cfg.eval.degs:
        deg = int(deg)
        for n in range(int(cfg.eval.n_per_deg)):
            az = torch.tensor([float(deg)], device=device)
            Xgen = heun_integrate(
                model, shape, az, int(cfg.eval.n_steps), device,
                fourier_feats=cfg.model.fourier_feats, t_dim=cfg.model.t_dim, az_source='deg',
            )  # [1,4,F,T]
            itd_g, ild_g = itd_ild_proxy(Xgen)
            # time domain
            L = istft_wave(Xgen[0,0:2], cfg.stft.n_fft, cfg.stft.hop, cfg.stft.window, length,
                           normalized=cfg.stft.normalized, win_length=cfg.stft.win_length)
            R = istft_wave(Xgen[0,2:4], cfg.stft.n_fft, cfg.stft.hop, cfg.stft.window, length,
                           normalized=cfg.stft.normalized, win_length=cfg.stft.win_length)
            l = L - L.mean(); r = R - R.mean()
            lag = gcc_phat_tdoa(l, r, max_lag)
            lat_pred = tdoa_to_lateral_deg(lag, cfg.data.sr)

            # Reference from templates
            ref = templates.get(deg, None)
            if ref is None:
                # fallback: use nearest degree template
                near_deg = min(templates.keys(), key=lambda k: min(abs(k-deg), 360-abs(k-deg)))
                ref = templates[near_deg]

            # Template-matching global azimuth prediction in SOFA frame
            w_itd = float(cfg.eval.template_weights.itd)
            w_ild = float(cfg.eval.template_weights.ild)
            feat_itd = float(itd_g[0].item())
            feat_ild = float(ild_g[0].item())
            best_deg = None
            best_score = float('inf')
            for k, v in templates.items():
                di = feat_itd - v["itd"]
                dl = feat_ild - v["ild"]
                score = (w_itd * di) ** 2 + (w_ild * dl) ** 2
                if score < best_score:
                    best_score = score
                    best_deg = int(k)
            az_pred_tm = best_deg if best_deg is not None else deg
            az_err_circ = circular_diff_deg(az_pred_tm, deg)
            lat_tgt = lateral_deg_from_global(deg)

            res = {
                "deg": deg,
                "itd_pred": float(itd_g[0].item()),
                "ild_pred": float(ild_g[0].item()),
                "lat_pred": float(lat_pred),
                "az_pred_tm": int(az_pred_tm),
                "az_circ_err": float(az_err_circ),
                "itd_ref": ref["itd"],
                "ild_ref": ref["ild"],
                "lat_ref": ref["lat"],
                "lat_tgt": float(lat_tgt),
                "iacc_ref": ref["iacc"],
                "itd_abs_err": abs(float(itd_g[0].item()) - ref["itd"]),
                "ild_abs_err": abs(float(ild_g[0].item()) - ref["ild"]),
                "lat_abs_err": abs(float(lat_pred) - float(lat_tgt)),
            }
            results.append(res)

    # Aggregate and save CSV
    out_csv = os.path.join(cfg.eval.out_dir, "eval_results.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader(); writer.writerows(results)
    print(f"Wrote eval metrics to {out_csv}  (N={len(results)})")

if __name__ == "__main__":
    run()
