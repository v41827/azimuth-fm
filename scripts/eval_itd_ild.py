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

try:
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAVE_PLOTTING = True
except Exception:
    HAVE_PLOTTING = False


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
            # IACC for generated
            nrm = torch.norm(l) * torch.norm(r) + 1e-8
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
            iacc_pred = float((xcorr[low:high].abs().max() / nrm).item())

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
                "iacc_pred": float(iacc_pred),
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

    # ----------------------------
    # Plots and summaries
    # ----------------------------
    if HAVE_PLOTTING:
        df = pd.DataFrame(results)
        plots_dir = os.path.join(cfg.eval.out_dir, "plots"); os.makedirs(plots_dir, exist_ok=True)

        # Aggregate by degree
        g = df.groupby("deg")
        agg = g.agg({
            "itd_pred":"mean", "ild_pred":"mean", "iacc_pred":"mean",
            "itd_ref":"mean",  "ild_ref":"mean",  "iacc_ref":"mean",
            "lat_abs_err":"mean", "az_circ_err":"mean"
        }).reset_index()
        agg.to_csv(os.path.join(plots_dir, "aggregates_by_deg.csv"), index=False)

        # ITD vs azimuth
        plt.figure(figsize=(6,4))
        plt.plot(agg["deg"], agg["itd_ref"], label="ref", linewidth=2)
        plt.plot(agg["deg"], agg["itd_pred"], label="model", linewidth=2)
        plt.xlabel("Azimuth (deg)"); plt.ylabel("ITD proxy (arb)"); plt.legend(); plt.tight_layout()
        p_itd = os.path.join(plots_dir, "itd_vs_az.png"); plt.savefig(p_itd, dpi=160); plt.close()

        # ILD vs azimuth
        plt.figure(figsize=(6,4))
        plt.plot(agg["deg"], agg["ild_ref"], label="ref", linewidth=2)
        plt.plot(agg["deg"], agg["ild_pred"], label="model", linewidth=2)
        plt.xlabel("Azimuth (deg)"); plt.ylabel("ILD (dB, proxy)"); plt.legend(); plt.tight_layout()
        p_ild = os.path.join(plots_dir, "ild_vs_az.png"); plt.savefig(p_ild, dpi=160); plt.close()

        # IACC vs azimuth
        plt.figure(figsize=(6,4))
        plt.plot(agg["deg"], agg["iacc_ref"], label="ref", linewidth=2)
        plt.plot(agg["deg"], agg["iacc_pred"], label="model", linewidth=2)
        plt.xlabel("Azimuth (deg)"); plt.ylabel("IACC"); plt.ylim(0,1.0); plt.legend(); plt.tight_layout()
        p_iacc = os.path.join(plots_dir, "iacc_vs_az.png"); plt.savefig(p_iacc, dpi=160); plt.close()

        # Lateral MAE vs azimuth
        plt.figure(figsize=(6,4))
        plt.plot(agg["deg"], agg["lat_abs_err"], label="lateral MAE", linewidth=2)
        plt.xlabel("Azimuth (deg)"); plt.ylabel("|lat_pred - lat_tgt| (deg)"); plt.tight_layout()
        p_latmae = os.path.join(plots_dir, "lateral_mae_vs_az.png"); plt.savefig(p_latmae, dpi=160); plt.close()

        # Circular azimuth MAE vs azimuth (SOFA frame)
        plt.figure(figsize=(6,4))
        plt.plot(agg["deg"], agg["az_circ_err"], label="azimuth circ. MAE", linewidth=2)
        plt.xlabel("Azimuth (deg)"); plt.ylabel("Circular MAE (deg)"); plt.tight_layout()
        p_azmae = os.path.join(plots_dir, "azimuth_circ_mae_vs_az.png"); plt.savefig(p_azmae, dpi=160); plt.close()

        # Confusion matrix (targets vs template-matched preds)
        cm = pd.crosstab(df["deg"], df["az_pred_tm"], normalize="index")
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, cmap="viridis", vmin=0, vmax=1)
        plt.xlabel("Predicted azimuth"); plt.ylabel("Target azimuth"); plt.tight_layout()
        p_cm = os.path.join(plots_dir, "confusion_matrix.png"); plt.savefig(p_cm, dpi=160); plt.close()

        # Scatter ITD and ILD
        plt.figure(figsize=(5,4))
        plt.scatter(df["itd_ref"], df["itd_pred"], s=6, alpha=0.6)
        lims = [df[["itd_ref","itd_pred"]].min().min(), df[["itd_ref","itd_pred"]].max().max()]
        plt.plot(lims, lims, 'r--'); plt.xlabel('ITD ref'); plt.ylabel('ITD pred'); plt.tight_layout()
        p_sc_itd = os.path.join(plots_dir, "scatter_itd.png"); plt.savefig(p_sc_itd, dpi=160); plt.close()

        plt.figure(figsize=(5,4))
        plt.scatter(df["ild_ref"], df["ild_pred"], s=6, alpha=0.6)
        lims = [df[["ild_ref","ild_pred"]].min().min(), df[["ild_ref","ild_pred"]].max().max()]
        plt.plot(lims, lims, 'r--'); plt.xlabel('ILD ref'); plt.ylabel('ILD pred'); plt.tight_layout()
        p_sc_ild = os.path.join(plots_dir, "scatter_ild.png"); plt.savefig(p_sc_ild, dpi=160); plt.close()

        print(f"Saved plots under {plots_dir}")

    # W&B logging (independent of plotting stack)
    try:
        if cfg.eval.wandb.enable:
            import wandb as _wb
            _wb.init(project=cfg.eval.wandb.project, config=OmegaConf.to_container(cfg, resolve=True))
            logs = {}
            # Log table if pandas available
            try:
                if 'pd' in globals():
                    df = pd.read_csv(out_csv)
                    logs["eval/table"] = _wb.Table(dataframe=df)
            except Exception:
                pass
            # Log images if they exist
            for name in [
                "itd_vs_az", "ild_vs_az", "iacc_vs_az",
                "lateral_mae", "azimuth_circ_mae",
                "confusion", "scatter_itd", "scatter_ild",
            ]:
                p = os.path.join(cfg.eval.out_dir, "plots", f"{name}.png")
                if os.path.isfile(p):
                    logs[f"eval/{name}"] = _wb.Image(p)
            if logs:
                _wb.log(logs)

            # Also upload CSV and plots dir as an Artifact
            try:
                art = _wb.Artifact("azfm-eval-results", type="eval")
                art.add_file(out_csv)
                plots_dir = os.path.join(cfg.eval.out_dir, "plots")
                if os.path.isdir(plots_dir):
                    art.add_dir(plots_dir)
                _wb.log_artifact(art)
            except Exception:
                pass
    except Exception as e:
        print(f"[EVAL] W&B logging failed: {e}")

    # -------------------------------------------------
    # Optional: dump per-angle example figures (gen vs ref)
    # -------------------------------------------------
    if bool(getattr(cfg.eval, "dump_examples", True)) and HAVE_PLOTTING:
        # Helper: find a reference sample in val set close to target degree
        ds_ref = BinauralSTFTDataset(
            cfg.data.val_csv, cfg.data.audio_root, sr=cfg.data.sr, seg_sec=cfg.data.seg_sec,
            n_fft=cfg.stft.n_fft, hop=cfg.stft.hop, win=cfg.stft.window,
            normalized=cfg.stft.normalized, win_length=cfg.stft.win_length,
            az_mode=cfg.data.az_mode, enforce_csv_sr=True,
        )
        def deg_from_az(az):
            if isinstance(az, torch.Tensor) and az.numel() == 2:
                return (math.degrees(math.atan2(float(az[0]), float(az[1]))) + 360.0) % 360.0
            return float(az) % 360.0

        def pick_ref_for_deg(target_deg: int):
            best_i, best_d = None, 1e9
            for i in range(min(len(ds_ref), 2000)):
                Xr, azr, Nr = ds_ref[i]
                d = circular_diff_deg(deg_from_az(azr), target_deg)
                if d < best_d:
                    best_i, best_d = (Xr, azr, Nr), d
                    if d < 0.5:
                        break
            return best_i

        ex_dir = os.path.join(cfg.eval.out_dir, "examples"); os.makedirs(ex_dir, exist_ok=True)
        ex_degs = cfg.eval.example_degs if cfg.eval.example_degs is not None else cfg.eval.degs
        for deg in ex_degs:
            deg = int(deg)
            ref = pick_ref_for_deg(deg)
            if ref is None:
                continue
            Xref, azr, Nr = ref
            # Generate N examples and plot the first one
            az = torch.tensor([float(deg)], device=device)
            Xgen = heun_integrate(
                model, shape, az, int(cfg.eval.n_steps), device,
                fourier_feats=cfg.model.fourier_feats, t_dim=cfg.model.t_dim, az_source='deg',
            )
            # Reconstruct time-domain
            Lg = istft_wave(Xgen[0,0:2], cfg.stft.n_fft, cfg.stft.hop, cfg.stft.window, length,
                            normalized=cfg.stft.normalized, win_length=cfg.stft.win_length)
            Rg = istft_wave(Xgen[0,2:4], cfg.stft.n_fft, cfg.stft.hop, cfg.stft.window, length,
                            normalized=cfg.stft.normalized, win_length=cfg.stft.win_length)
            Lr = istft_wave(Xref[0:2], cfg.stft.n_fft, cfg.stft.hop, cfg.stft.window, int(Nr),
                            normalized=cfg.stft.normalized, win_length=cfg.stft.win_length)
            Rr = istft_wave(Xref[2:4], cfg.stft.n_fft, cfg.stft.hop, cfg.stft.window, int(Nr),
                            normalized=cfg.stft.normalized, win_length=cfg.stft.win_length)

            # Spectrograms (magnitude dB)
            def mag_db(X):
                c = torch.complex(X[0], X[1])
                m = (c.abs() + 1e-7).cpu().numpy()
                return 20.0 * np.log10(m)
            specL_gen = mag_db(Xgen[0,0:2]); specR_gen = mag_db(Xgen[0,2:4])
            specL_ref = mag_db(Xref[0:2]);   specR_ref = mag_db(Xref[2:4])
            vmin, vmax = -80, 0
            plt.figure(figsize=(9,6))
            for i,(S,title) in enumerate([
                (specL_ref, 'Ref L'), (specL_gen, 'Gen L'),
                (specR_ref, 'Ref R'), (specR_gen, 'Gen R')
            ]):
                ax = plt.subplot(2,2,i+1)
                im = ax.imshow(S, aspect='auto', origin='lower', vmin=vmin, vmax=vmax, cmap='magma')
                ax.set_title(title)
                ax.set_xlabel('T'); ax.set_ylabel('F')
            plt.tight_layout(); p = os.path.join(ex_dir, f"deg{deg:03d}_spec.png"); plt.savefig(p, dpi=160); plt.close()

            # IPD heatmap (low bands)
            def ipd_heat(X):
                dph = np.angle(np.exp(1j*(X[0].cpu().numpy().astype(np.float64) - X[2].cpu().numpy().astype(np.float64))))
                return dph
            ipd_ref = ipd_heat(torch.stack([Xref[0], Xref[1], Xref[2], Xref[3]]))
            ipd_gen = ipd_heat(Xgen[0])
            lowF = slice(0, min(40, ipd_ref.shape[0]))
            plt.figure(figsize=(8,4))
            ax = plt.subplot(1,2,1); im=ax.imshow(ipd_ref[lowF], aspect='auto', origin='lower', vmin=-np.pi, vmax=np.pi, cmap='twilight'); ax.set_title('Ref IPD')
            ax = plt.subplot(1,2,2); im=ax.imshow(ipd_gen[lowF], aspect='auto', origin='lower', vmin=-np.pi, vmax=np.pi, cmap='twilight'); ax.set_title('Gen IPD')
            plt.tight_layout(); p = os.path.join(ex_dir, f"deg{deg:03d}_ipd.png"); plt.savefig(p, dpi=160); plt.close()

            # ILD curves (avg over time)
            def ild_curve(X):
                Lc = np.abs((X[0].cpu().numpy() + 1j*X[1].cpu().numpy()))
                Rc = np.abs((X[2].cpu().numpy() + 1j*X[3].cpu().numpy()))
                ild = 20*np.log10((Lc+1e-7)/(Rc+1e-7))
                return ild.mean(axis=1)
            ild_ref = ild_curve(Xref); ild_gen = ild_curve(Xgen[0])
            plt.figure(figsize=(6,4))
            plt.plot(ild_ref, label='Ref'); plt.plot(ild_gen, label='Gen'); plt.legend(); plt.xlabel('F bin'); plt.ylabel('ILD (dB)'); plt.tight_layout()
            p = os.path.join(ex_dir, f"deg{deg:03d}_ild_curve.png"); plt.savefig(p, dpi=160); plt.close()

            # Waveforms overlay
            t = np.arange(Lg.numel())/cfg.data.sr
            plt.figure(figsize=(8,4))
            plt.subplot(2,1,1); plt.plot(t, Lr.cpu().numpy(), label='Ref L', alpha=0.7); plt.plot(t, Lg.cpu().numpy(), label='Gen L', alpha=0.7); plt.legend(); plt.ylabel('amp')
            plt.subplot(2,1,2); plt.plot(t, Rr.cpu().numpy(), label='Ref R', alpha=0.7); plt.plot(t, Rg.cpu().numpy(), label='Gen R', alpha=0.7); plt.legend(); plt.xlabel('s'); plt.ylabel('amp')
            plt.tight_layout(); p = os.path.join(ex_dir, f"deg{deg:03d}_wave.png"); plt.savefig(p, dpi=160); plt.close()

        # Optional W&B upload of examples directory
        try:
            if cfg.eval.wandb.enable:
                import wandb as _wb
                art = _wb.Artifact('azfm-eval-examples', type='eval_examples')
                art.add_dir(ex_dir)
                _wb.log_artifact(art)
        except Exception:
            pass

if __name__ == "__main__":
    run()
