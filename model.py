import torch, torch.nn as nn

def cbr(i, o, s):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.GroupNorm(8, o), nn.GELU())

class Encoder(nn.Module):
    def __init__(self, in_ch, w=32):
        super().__init__()
        self.s1 = nn.Sequential(cbr(in_ch, w, 1), cbr(w, w, 2))
        self.s2 = nn.Sequential(cbr(w, 2 * w, 1), cbr(2 * w, 2 * w, 2))
        self.s3 = nn.Sequential(cbr(2 * w, 4 * w, 1), cbr(4 * w, 4 * w, 2))
        self.out_dim = 4 * w
    def forward(self, x):
        f = self.s3(self.s2(self.s1(x)))          # (B, 4w, 4, 4)
        return f, f.mean((2, 3))

class Decoder(nn.Module):
    def __init__(self, c_in, out_ch):
        super().__init__()
        def up(i, o): return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.GroupNorm(8, o), nn.GELU())
        self.net = nn.Sequential(up(c_in, c_in // 2), up(c_in // 2, c_in // 4), up(c_in // 4, c_in // 4),
                                 nn.Conv2d(c_in // 4, out_ch, 1))
    def forward(self, f): return self.net(f)

class HydroNet(nn.Module):
    def __init__(self, in_ch, seq_dim, use_temporal=True, width=32, emb=128, drop=0.3):
        super().__init__()
        self.enc = Encoder(in_ch, width)
        self.use_temporal = use_temporal
        fdim = self.enc.out_dim
        if use_temporal:
            self.gru = nn.GRU(seq_dim, 64, batch_first=True)
            fdim += 64
        self.fuse = nn.Sequential(nn.Linear(fdim, emb), nn.GELU(), nn.Dropout(drop))
        self.well_head = nn.Linear(emb, 2)            # [z regression, stressed logit]
        self.att_V, self.att_U = nn.Linear(emb, 64), nn.Linear(emb, 64)
        self.att_w = nn.Linear(64, 1)
        self.att_drop = nn.Dropout(0.25)
        self.mil_head = nn.Linear(emb, 1)             # linear => bag logit = sum(attn * instance logit)

    def embed(self, x, seq):
        _, p = self.enc(x)
        if self.use_temporal:
            _, hn = self.gru(seq)
            p = torch.cat([p, hn[-1]], 1)
        return self.fuse(p)

    def well_forward(self, h):
        o = self.well_head(h)
        return o[:, 0], o[:, 1]

    def mil_forward(self, h):                          # h: (N, emb) = one bag
        a = self.att_w(self.att_drop(torch.tanh(self.att_V(h)) * torch.sigmoid(self.att_U(h)))).squeeze(-1)
        a = torch.softmax(a, 0)
        inst = self.mil_head(h).squeeze(-1)
        return (a * inst).sum(), a, inst

class UncertaintyWeighting(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.s = nn.Parameter(torch.zeros(n))
    def forward(self, losses, ids):
        s = self.s.clamp(-3, 3)
        return sum(torch.exp(-s[i]) * l + s[i] for i, l in zip(ids, losses))
