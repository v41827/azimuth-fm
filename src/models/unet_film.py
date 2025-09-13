import torch, torch.nn as nn, torch.nn.functional as F

class FiLM(nn.Module):
    def __init__(self, cond_dim, n_channels):
        super().__init__()
        self.lin = nn.Linear(cond_dim, 2 * n_channels)
    def forward(self, x, h):
        gb = self.lin(h)[:, :, None, None]
        gamma, beta = gb.chunk(2, dim=1)
        return x * (1 + gamma) + beta

class Block(nn.Module):
    def __init__(self, c_in, c_out, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, c_out)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.film1 = FiLM(cond_dim, c_out)
        self.film2 = FiLM(cond_dim, c_out)
        self.skip = (c_in != c_out)
        if self.skip: self.proj = nn.Conv2d(c_in, c_out, 1)
    def forward(self, x, h):
        s = self.proj(x) if self.skip else x
        x = self.conv1(x); x = self.norm1(x); x = F.silu(x); x = self.film1(x, h)
        x = self.conv2(x); x = self.norm2(x); x = F.silu(x); x = self.film2(x, h)
        return x + s

class TinyUNet(nn.Module):
    def __init__(self, base=64, depth=3, cond_dim=128):
        super().__init__()
        self.in_conv = nn.Conv2d(4, base, 3, padding=1)
        downs, ups = [], []
        ch = base
        for _ in range(depth):
            downs.append(Block(ch, ch*2, cond_dim)); ch *= 2
        self.downs = nn.ModuleList(downs)
        self.mid = Block(ch, ch, cond_dim)
        for _ in range(depth):
            # After upsample we concat with skip of the same channel count (ch)
            # so the input to the up block is 2*ch, and we reduce to ch//2.
            ups.append(Block(ch * 2, ch // 2, cond_dim)); ch //= 2
        self.ups = nn.ModuleList(ups)
        self.out_conv = nn.Conv2d(base, 4, 3, padding=1)
        self.pools = nn.ModuleList([nn.AvgPool2d(2) for _ in range(depth)])
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.mlp = nn.Sequential(nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
    def forward(self, x, h):
        h = self.mlp(h)
        feats = []
        x = self.in_conv(x)
        for i, blk in enumerate(self.downs):
            x = blk(x, h); feats.append(x)
            x = self.pools[i](x)
        x = self.mid(x, h)
        for i, blk in enumerate(self.ups):
            x = self.upsample(x)
            skip = feats[-(i+1)]
            dh = skip.shape[-2] - x.shape[-2]; dw = skip.shape[-1] - x.shape[-1]
            if dh or dw: x = F.pad(x, (0, max(0,dw), 0, max(0,dh)))
            x = torch.cat([x, skip], dim=1)
            x = blk(x, h)
        return self.out_conv(x)
