import torch

@torch.no_grad()
def itd_ild_proxy(X4: torch.Tensor):
    B, C, F, T = X4.shape
    Lc = torch.complex(X4[:,0], X4[:,1]); Rc = torch.complex(X4[:,2], X4[:,3])
    magL = (Lc.abs() + 1e-7); magR = (Rc.abs() + 1e-7)
    phaseL = torch.atan2(X4[:,1], X4[:,0]); phaseR = torch.atan2(X4[:,3], X4[:,2])
    low = slice(0, min(15, F))
    # Phase difference wrapped to [-pi, pi]
    d = phaseL[:, low, :] - phaseR[:, low, :]
    dphi = torch.atan2(torch.sin(d), torch.cos(d))
    itd_proxy = dphi.mean(dim=(1,2))
    hf = slice(int(F * 0.4), F)
    ild_db = (20 * torch.log10(magL[:, hf, :].mean(dim=(1,2)) / magR[:, hf, :].mean(dim=(1,2)))).clamp(-30, 30)
    return itd_proxy, ild_db

def corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-8))
