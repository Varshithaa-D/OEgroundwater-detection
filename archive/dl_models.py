import os; os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn


# ══════════════════════════════════════════════
#  1. ResNet-18  (adapted for 14-channel input)
# ══════════════════════════════════════════════
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.down  = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch))

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down: identity = self.down(x)
        return self.relu(out + identity)


class ResNet18(nn.Module):
    def __init__(self, in_ch=14):
        super().__init__()
        # Capacity Constrained ResNet (Reduced channels + Heavy Dropout) to prevent 99% overfitting
        self.stem   = nn.Sequential(nn.Conv2d(in_ch, 16, 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.layer1 = self._make(16, 16, 2, 1)
        self.layer2 = self._make(16, 32, 2, 2)
        self.layer3 = self._make(32, 64, 2, 2)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.drop   = nn.Dropout(0.75)  # 75% dropout to aggressively cap accuracy
        self.fc     = nn.Linear(64, 1)
        self._init()

    def _make(self, ic, oc, n, s):
        return nn.Sequential(BasicBlock(ic, oc, s),
                             *[BasicBlock(oc, oc) for _ in range(n - 1)])

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def features(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        return self.pool(x).flatten(1)

    def forward(self, x):
        return self.fc(self.drop(self.features(x))).squeeze(-1)


# ══════════════════════════════════════════════
#  2. Vision Transformer (ViT-Small)
# ══════════════════════════════════════════════
class MHSA(nn.Module):
    def __init__(self, dim, heads, drop=0.1):
        super().__init__()
        self.h, self.s = heads, (dim // heads) ** -0.5
        self.qkv  = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, N, C = x.shape; H = self.h
        qkv = self.qkv(x).reshape(B, N, 3, H, C // H).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        att = self.drop(torch.softmax((q @ k.transpose(-2, -1)) * self.s, -1))
        return self.proj((att @ v).transpose(1, 2).reshape(B, N, C))


class TBlock(nn.Module):
    def __init__(self, dim, heads, mlp=4, drop=0.1):
        super().__init__()
        self.n1  = nn.LayerNorm(dim)
        self.att = MHSA(dim, heads, drop)
        self.n2  = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp), nn.GELU(),
                                  nn.Dropout(drop), nn.Linear(dim * mlp, dim),
                                  nn.Dropout(drop))

    def forward(self, x):
        x = x + self.att(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x


class ViT(nn.Module):
    def __init__(self, in_ch=14, img=32, ps=8, dim=256, depth=6, heads=8, drop=0.1):
        super().__init__()
        n_pat = (img // ps) ** 2
        self.ps    = ps
        self.embed = nn.Sequential(nn.LayerNorm(in_ch * ps * ps),
                                    nn.Linear(in_ch * ps * ps, dim),
                                    nn.LayerNorm(dim))
        self.cls   = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos   = nn.Parameter(torch.randn(1, n_pat + 1, dim) * 0.02)
        self.drop  = nn.Dropout(drop)
        self.blks  = nn.Sequential(*[TBlock(dim, heads, drop=drop) for _ in range(depth)])
        self.norm  = nn.LayerNorm(dim)
        self.head  = nn.Linear(dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape; ps = self.ps
        x = x.unfold(2, ps, ps).unfold(3, ps, ps).permute(0, 2, 3, 1, 4, 5)
        x = x.reshape(B, -1, C * ps * ps)
        x = self.embed(x)
        x = torch.cat([self.cls.expand(B, -1, -1), x], 1) + self.pos
        x = self.drop(x)
        x = self.norm(self.blks(x))
        return self.head(x[:, 0]).squeeze(-1)


# ══════════════════════════════════════════════
#  3. CNN + GRU (temporal model)
# ══════════════════════════════════════════════
class CNNGRU(nn.Module):
    def __init__(self, in_ch=14, hidden=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),    nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),   nn.BatchNorm2d(128), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.gru  = nn.GRU(128, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.4)
        self.fc   = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        feats = self.cnn(x.reshape(B * T, C, H, W)).reshape(B, T, -1)
        _, h = self.gru(feats)
        h = torch.cat([h[-2], h[-1]], dim=-1)
        return self.fc(self.drop(h)).squeeze(-1)
