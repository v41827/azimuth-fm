import torch

def stft_wave(x, n_fft: int, hop: int, win: str = "hann"):
    window = torch.hann_window(n_fft, device=x.device) if win == "hann" else None
    X = torch.stft(x, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
    return torch.stack([X.real, X.imag], dim=1)

def istft_wave(Xri, n_fft: int, hop: int, win: str, length: int):
    window = torch.hann_window(n_fft, device=Xri.device) if win == "hann" else None
    # Xri: [2, F, T] → complex [F, T]
    X = torch.complex(Xri[0], Xri[1])
    return torch.istft(X, n_fft=n_fft, hop_length=hop, window=window, length=length)
