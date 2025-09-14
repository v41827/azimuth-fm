import os, time, torch, torch.nn.functional as F
from contextlib import nullcontext
import torchaudio
import torchaudio
from torch.utils.data import DataLoader
from hydra import main
from omegaconf import DictConfig, OmegaConf
from src.utils.seed import set_seed
from src.data.dataset import BinauralSTFTDataset
from src.models.unet_film import TinyUNet
from src.fm.objectives import linear_path_xt, target_velocity_const
from src.fm.cond import build_cond
from src.utils.metrics import itd_ild_proxy, corr
from src.utils.audio import istft_wave
from src.utils.audio import istft_wave

try:
    import wandb
    HAVE_WANDB = True
except Exception:
    HAVE_WANDB = False

@main(config_path="../configs", config_name="config", version_base="1.3")
def run(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    os.makedirs(cfg.train.out_dir, exist_ok=True)
    set_seed(cfg.train.seed)

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    train_set = BinauralSTFTDataset(
        cfg.data.train_csv, cfg.data.audio_root, sr=cfg.data.sr,
        seg_sec=cfg.data.seg_sec, n_fft=cfg.stft.n_fft, hop=cfg.stft.hop,
        win=cfg.stft.window, normalized=cfg.stft.normalized, win_length=cfg.stft.win_length,
        az_mode=cfg.data.az_mode, enforce_csv_sr=True,
    )
    val_set = BinauralSTFTDataset(
        cfg.data.val_csv, cfg.data.audio_root, sr=cfg.data.sr,
        seg_sec=cfg.data.seg_sec, n_fft=cfg.stft.n_fft, hop=cfg.stft.hop,
        win=cfg.stft.window, normalized=cfg.stft.normalized, win_length=cfg.stft.win_length,
        az_mode=cfg.data.az_mode, enforce_csv_sr=True,
    )

    pw = bool(getattr(cfg.train, "num_workers", 0) and cfg.train.num_workers > 0)
    train_loader = DataLoader(train_set, batch_size=cfg.train.batch_size, shuffle=True,
                              num_workers=cfg.train.num_workers, drop_last=True,
                              pin_memory=True, persistent_workers=pw)
    val_loader   = DataLoader(val_set,   batch_size=cfg.train.batch_size, shuffle=False,
                              num_workers=cfg.train.num_workers, drop_last=True,
                              pin_memory=True, persistent_workers=pw)

    # infer shapes & cond_dim
    X0, _, _ = next(iter(train_loader))
    nF, nT = X0.shape[-2], X0.shape[-1]
    shape = (4, nF, nT)
    cond_dim = 2 * cfg.model.fourier_feats + cfg.model.t_dim

    model = TinyUNet(base=cfg.model.base_ch, depth=cfg.model.depth, cond_dim=cond_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(getattr(cfg.train, "amp", True)))

    if cfg.train.wandb.enable and HAVE_WANDB:
        wandb.init(project=cfg.train.wandb.project, config=OmegaConf.to_container(cfg, resolve=True))

    gstep = 0
    start = time.time()
    model.train()
    while gstep < cfg.train.steps:
        for X, az, _ in train_loader:
            gstep += 1
            X = X.to(device)
            az = torch.as_tensor(az, device=device, dtype=torch.float32)
            x0 = torch.randn_like(X); x1 = X
            t = torch.rand((X.shape[0],), device=device)
            xt = linear_path_xt(x0, x1, t)
            vstar = target_velocity_const(x0, x1)
            if cfg.data.az_mode == "sincos" and az.ndim == 2:
                h = build_cond(
                    t,
                    az_source="sincos",
                    az_sin=az[:, 0],
                    az_cos=az[:, 1],
                    fourier_feats=cfg.model.fourier_feats,
                    t_dim=cfg.model.t_dim,
                )
            else:
                h = build_cond(
                    t,
                    az_source="deg",
                    az_deg=az,
                    fourier_feats=cfg.model.fourier_feats,
                    t_dim=cfg.model.t_dim,
                )

            amp_ctx = (
                torch.amp.autocast("cuda", enabled=scaler.is_enabled())
                if device.type == "cuda"
                else nullcontext()
            )
            with amp_ctx:
                vhat = model(xt, h)
                loss = F.mse_loss(vhat, vstar)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if float(getattr(cfg.train, "grad_clip", 0.0)) > 0.0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(opt)
            scaler.update()

            if gstep % 100 == 0:
                itps = gstep / (time.time() - start + 1e-6)
                print(f"[{gstep}/{cfg.train.steps}] loss={loss.item():.4f} ~{itps:.2f} it/s")
                if cfg.train.wandb.enable and HAVE_WANDB:
                    wandb.log({"train/loss": float(loss.item()), "step": gstep, "itps": itps})

            if gstep % cfg.train.val_every == 0:
                validate(cfg, model, val_loader, device, shape, gstep)
                model.train()

            if gstep % cfg.train.save_every == 0:
                ckpt = {"model": model.state_dict(), "cfg": OmegaConf.to_container(cfg, resolve=True)}
                ckpt_path = os.path.join(cfg.train.out_dir, f"ckpt_{gstep}.pt")
                torch.save(ckpt, ckpt_path)
                # Optionally upload checkpoint as W&B Artifact
                if (
                    cfg.train.wandb.enable and HAVE_WANDB and
                    bool(getattr(cfg.train.wandb, "upload_artifacts", False)) and
                    (gstep % int(getattr(cfg.train.wandb, "artifact_ckpt_every", cfg.train.save_every)) == 0)
                ):
                    try:
                        import wandb as _wb
                        prefix = getattr(cfg.train.wandb, "artifact_prefix", "azfm")
                        art = _wb.Artifact(f"{prefix}-ckpt-{gstep}", type="model", metadata={"step": int(gstep)})
                        art.add_file(ckpt_path)
                        _wb.log_artifact(art)
                    except Exception as e:
                        print(f"[ARTIFACT] ckpt upload failed at step {gstep}: {e}")

            if gstep >= cfg.train.steps:
                break

    final_path = os.path.join(cfg.train.out_dir, "ckpt_final.pt")
    torch.save({"model": model.state_dict(), "cfg": OmegaConf.to_container(cfg, resolve=True)}, final_path)
    if cfg.train.wandb.enable and HAVE_WANDB and bool(getattr(cfg.train.wandb, "upload_artifacts", False)):
        try:
            import wandb as _wb
            prefix = getattr(cfg.train.wandb, "artifact_prefix", "azfm")
            art = _wb.Artifact(f"{prefix}-ckpt-final", type="model", metadata={"step": int(cfg.train.steps)})
            art.add_file(final_path)
            _wb.log_artifact(art)
        except Exception as e:
            print(f"[ARTIFACT] final ckpt upload failed: {e}")
    print("Done.")

@torch.inference_mode()
def validate(cfg, model, val_loader, device, shape, step):
    model.eval()
    X, az, _ = next(iter(val_loader))
    X = X.to(device); az = torch.as_tensor(az, device=device, dtype=torch.float32)
    from src.fm.integrators import heun_integrate
    Xgen = heun_integrate(
        model,
        shape,
        az,
        cfg.train.sample_steps,
        device,
        fourier_feats=cfg.model.fourier_feats,
        t_dim=cfg.model.t_dim,
        az_source=("sincos" if (cfg.data.az_mode == "sincos" and az.ndim == 2) else "deg"),
    )
    itd_gt, ild_gt = itd_ild_proxy(X); itd_pr, ild_pr = itd_ild_proxy(Xgen)
    c_itd = corr(itd_gt, itd_pr); c_ild = corr(ild_gt, ild_pr)
    print(f"[VAL {step}] ITD corr: {c_itd:.4f}  ILD corr: {c_ild:.4f}")
    if cfg.train.wandb.enable and HAVE_WANDB:
        wandb.log({"val/itd_corr": c_itd, "val/ild_corr": c_ild, "step": step})

    # Dump user-defined azimuth samples and optionally log to W&B
    if bool(getattr(cfg.train, "val_dump_audio", True)):
        try:
            from src.fm.integrators import heun_integrate as _heun
            deg_list = list(getattr(cfg.train, "val_sample_degs", [0, 30, 60, 90, 330]))
            steps_list = list(getattr(cfg.train, "val_show_steps", [cfg.train.sample_steps]))
            length = int(cfg.data.seg_sec * cfg.data.sr)
            base_dir = os.path.join(cfg.train.out_dir, f"val_samples/step_{step}")
            for s in steps_list:
                out_dir = os.path.join(base_dir, f"s{s}")
                os.makedirs(out_dir, exist_ok=True)
                degs = torch.tensor(deg_list, device=device, dtype=torch.float32)
                Xsyn = _heun(
                    model, shape, degs, int(s), device,
                    fourier_feats=cfg.model.fourier_feats, t_dim=cfg.model.t_dim, az_source="deg",
                )
                wb_audios = []
                for i, d in enumerate(deg_list):
                    L = istft_wave(
                        Xsyn[i, 0:2], cfg.stft.n_fft, cfg.stft.hop, cfg.stft.window, length,
                        normalized=cfg.stft.normalized, win_length=cfg.stft.win_length,
                    )
                    R = istft_wave(
                        Xsyn[i, 2:4], cfg.stft.n_fft, cfg.stft.hop, cfg.stft.window, length,
                        normalized=cfg.stft.normalized, win_length=cfg.stft.win_length,
                    )
                    wav = torch.stack([L, R], dim=0).cpu()
                    torchaudio.save(os.path.join(out_dir, f"pred_az{int(d)}.wav"), wav, cfg.data.sr)
                    if cfg.train.wandb.enable and HAVE_WANDB and bool(getattr(cfg.train.wandb, "log_audio", False)):
                        try:
                            import wandb
                            wb_audios.append(wandb.Audio(wav.numpy(), sample_rate=cfg.data.sr, caption=f"s{s}_az{int(d)}"))
                        except Exception:
                            pass
                if wb_audios and cfg.train.wandb.enable and HAVE_WANDB:
                    max_n = int(getattr(cfg.train.wandb, "max_audio", 10))
                    wandb.log({f"val/audio_s{s}": wb_audios[:max_n], "step": step})
                # Optionally upload this step's audio directory as a W&B Artifact
                if (
                    cfg.train.wandb.enable and HAVE_WANDB and
                    bool(getattr(cfg.train.wandb, "upload_artifacts", False)) and
                    (int(step) % int(getattr(cfg.train.wandb, "artifact_audio_every", 1000)) == 0)
                ):
                    try:
                        import wandb as _wb
                        prefix = getattr(cfg.train.wandb, "artifact_prefix", "azfm")
                        art = _wb.Artifact(f"{prefix}-val-audio-{int(step)}", type="audio", metadata={"step": int(step)})
                        art.add_dir(base_dir)
                        _wb.log_artifact(art)
                    except Exception as e:
                        print(f"[ARTIFACT] val audio upload failed: {e}")
        except Exception as e:
            print(f"[VAL {step}] audio dump failed: {e}")

if __name__ == "__main__":
    run()
