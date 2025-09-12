import os, math
from typing import Tuple, Union

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

from ..utils.audio import stft_wave


class BinauralSTFTDataset(Dataset):
    """
    Stereo WAVs already at desired sr.

    CSV can be either of the forms:
      - minimal:  filepath | azimuth
      - extended: filename | split | azimuth | azimuth_rad | az_sin | az_cos | samplerate | render_samplerate | hrir_id

    You can choose what the dataset returns for the azimuth via `az_mode`:
      - "deg" (default): returns scalar degrees (float)
      - "sincos": returns a 2-D tensor [sin, cos] taken directly from CSV

    If the CSV contains a `samplerate` column, it will be checked against `sr` (unless
    `enforce_csv_sr=False`).
    """

    def __init__(
        self,
        csv_path: str,
        audio_root: str,
        sr: int,
        seg_sec: float,
        n_fft: int,
        hop: int,
        win: str = "hann",
        az_mode: str = "deg",
        enforce_csv_sr: bool = True,
    ):
        self.df = pd.read_csv(csv_path)
        self.audio_root = audio_root
        self.sr = sr
        self.seg_sec = seg_sec
        self.n_fft = n_fft
        self.hop = hop
        self.win = win
        self.az_mode = az_mode
        self.enforce_csv_sr = enforce_csv_sr

        # basic column checks
        has_filename = "filename" in self.df.columns
        has_filepath = "filepath" in self.df.columns
        assert has_filename or has_filepath, "CSV must contain either 'filename' or 'filepath' column"
        assert "azimuth" in self.df.columns or ("az_sin" in self.df.columns and "az_cos" in self.df.columns), (
            "CSV must contain 'azimuth' (deg) or both 'az_sin' and 'az_cos' columns"
        )

        if self.enforce_csv_sr and "samplerate" in self.df.columns:
            # Warn in __init__ if any row disagrees with `sr`
            bad = self.df["samplerate"].astype(int) != int(self.sr)
            if bool(bad.any()):
                first_bad = self.df[bad].iloc[0]
                raise AssertionError(
                    f"CSV samplerate {first_bad['samplerate']} does not match expected sr={self.sr} for file {first_bad.get('filename', first_bad.get('filepath'))}"
                )

    def __len__(self) -> int:
        return len(self.df)

    def _load(self, row) -> Tuple[torch.Tensor, int]:
        key = "filepath" if "filepath" in self.df.columns else "filename"
        fp = row[key]
        path = fp if os.path.isabs(fp) else os.path.join(self.audio_root, fp)
        wav, sr = torchaudio.load(path)  # [C, N]
        return wav, sr

    def _ensure_len(self, wav: torch.Tensor, target_N: int) -> torch.Tensor:
        C, N = wav.shape
        if N == target_N:
            return wav
        if N > target_N:
            start = (N - target_N) // 2
            return wav[:, start : start + target_N]
        reps = math.ceil(target_N / N)
        return wav.repeat(1, reps)[:, :target_N]

    def _read_az(self, row) -> Union[float, torch.Tensor]:
        if self.az_mode == "sincos":
            # Prefer CSV columns; fall back to computing if absent
            if "az_sin" in self.df.columns and "az_cos" in self.df.columns:
                return torch.tensor([float(row["az_sin"]), float(row["az_cos"])], dtype=torch.float32)
            # Fallback: compute from degrees in CSV
            assert "azimuth" in self.df.columns, "Need 'azimuth' column to compute sin/cos"
            az = float(row["azimuth"]) % 360.0
            rad = az * math.pi / 180.0
            return torch.tensor([math.sin(rad), math.cos(rad)], dtype=torch.float32)
        else:
            # default: return scalar degrees
            assert "azimuth" in self.df.columns, "az_mode='deg' requires 'azimuth' column"
            return float(row["azimuth"]) % 360.0

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        wav, sr = self._load(row)
        # Hard fail if the actual file SR is wrong
        assert sr == self.sr, f"Expected {self.sr} Hz, got {sr} for index {idx}"

        target_N = int(self.seg_sec * sr)
        wav = self._ensure_len(wav, target_N)  # [2, N]

        # STFT per ear -> Re/Im channels
        L = stft_wave(wav[0:1, :], self.n_fft, self.hop, self.win)[0]
        R = stft_wave(wav[1:2, :], self.n_fft, self.hop, self.win)[0]
        X = torch.stack([L[0], L[1], R[0], R[1]], dim=0)  # [4, F, T]

        az = self._read_az(row)
        return X, az, target_N