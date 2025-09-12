import os, random, numpy as np, torch

def set_seed(seed: int | None, deterministic: bool = False):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    os.environ.setdefault("PYTHONHASHSEED", str(seed))