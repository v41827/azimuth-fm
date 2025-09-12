import os, torch, torchaudio
from torch.utils.data import DataLoader
from hydra import main
from omegaconf import DictConfig, OmegaConf
from src.data.dataset import BinauralSTFTDataset
from src.models.unet_film import TinyUNet
from src.fm.integrators import heun_integrate
from src.utils.audio import istft_wave

@main(config_path="../configs", config_name="config", version_base="1.3")
def run(cfg: DictConfig):
    args = cfg  # alias
    os.makedirs(cfg.sample.out_dir, exist_ok=True)
    device = torch.device(cfg.sample.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(cfg.sample.ckpt, map_location="cpu")
    cond_dim = 2 * cfg.model.fourier_feats + cfg.model.t_dim
    # model hyperparams from cfg (ckpt also saved cfg if you prefer restoring from that)
    model = TinyUNet(base=cfg.model.base_ch, depth=cfg.model.depth, cond_dim=cond_dim).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    ds = BinauralSTFTDataset(
        cfg.data.val_csv,
        cfg.data.audio_root,
        sr=cfg.data.sr,
        seg_sec=cfg.data.seg_sec,
        n_fft=cfg.stft.n_fft,
        hop=cfg.stft.hop,
        win=cfg.stft.window,
        az_mode=cfg.data.az_mode,
    )
    dl = DataLoader(ds, batch_size=cfg.sample.batch_size, shuffle=False, num_workers=2)

    X, az, _ = next(iter(dl))
    X = X.to(device); az = torch.tensor(az, device=device, dtype=torch.float32)
    F, T = X.shape[-2], X.shape[-1]
    shape = (4, F, T)

    Xgen = heun_integrate(
        model,
        shape,
        az,
        cfg.sample.n_steps,
        device,
        fourier_feats=cfg.model.fourier_feats,
        t_dim=cfg.model.t_dim,
        az_source=("sincos" if (cfg.data.az_mode == "sincos" and az.ndim == 2) else "deg"),
    )

    length = int(cfg.data.seg_sec * cfg.data.sr)
    out_dir = os.path.join(cfg.sample.out_dir, "audio"); os.makedirs(out_dir, exist_ok=True)
    B = min(4, Xgen.shape[0])
    for i in range(B):
        L = istft_wave(Xgen[i,0:2], cfg.stft.n_fft, cfg.stft.hop, cfg.stft.window, length)
        R = istft_wave(Xgen[i,2:4], cfg.stft.n_fft, cfg.stft.hop, cfg.stft.window, length)
        wav = torch.stack([L, R], dim=0).cpu()
        # Name file using azimuth degrees when possible
        try:
            if az.ndim == 2 and az.shape[1] == 2:
                import math
                deg = (math.degrees(math.atan2(float(az[i,0]), float(az[i,1]))) + 360.0) % 360.0
            else:
                deg = float(az[i].item())
        except Exception:
            deg = 0.0
        import torchaudio
        torchaudio.save(os.path.join(out_dir, f"pred_{i}_az{int(round(deg))}.wav"), wav, cfg.data.sr)
    print(f"Wrote {B} samples to {out_dir}")

if __name__ == "__main__":
    run()
