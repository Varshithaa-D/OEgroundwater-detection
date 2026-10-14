"""
stage3_deep_learning.py - Deep Sequence Models (Bi-GRU & Temporal 1D-CNN)
==========================================================================
Trains and evaluates end-to-end deep learning sequence architectures on 10-year
multi-spectral satellite sequences (2016-2025) under 25.0 km Spatially Buffered LOTO.

Architectures:
  1. DL_Bi_GRU: Bidirectional Gated Recurrent Unit (hidden 16) with LayerNorm & MLP head.
  2. DL_1D_CNN: Temporal 1D Convolutional Neural Network (kernel=3, 32 channels) learning
     multi-year local moving trends and depletion curvature.
"""

import os, json, random, argparse, warnings, time
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception as e:
    raise ImportError(f"PyTorch is required for Stage 3: {e}")

# ----------------------------------------------------------------------------- CONFIG
NPZ_PATH = "taluk_temporal_patches.npz"
INDEX_PATH = "index.parquet"
GPKG_PATH = "labeled_taluks.gpkg"
GPKG_TABLE, GPKG_KEY = "labeled_taluks", "key"
TALUK_COL, LABEL_COL = "taluk_id", "taluk_label"
N_YEARS, N_BANDS = 10, 13
SEED = 42
BUFFER_KM = 25.0
REF_LAT = 15.0
EPOCHS = 15
BATCH = 256
LR, WD = 1e-3, 1e-2
NOISE = 0.05
OUT_DIR = os.path.join("runs", "benchmarks")

def load_data(extra_neg=(-1,)):
    if os.path.exists(NPZ_PATH):
        d = np.load(NPZ_PATH, allow_pickle=True)
        mask = np.isin(d["labels"], [0, 1] + list(extra_neg))
        X = d["X"][mask].astype(np.float32)
        X[~np.isfinite(X)] = np.nan
        raw_labels = d["labels"][mask].astype(int)
        taluk_ids = d["taluk_ids"][mask].astype(str)
    else:
        seq_cols = [f"seq_y{y}_b{b}" for y in range(N_YEARS) for b in range(N_BANDS)]
        df = pd.read_parquet(INDEX_PATH, columns=[TALUK_COL, LABEL_COL] + seq_cols)
        df = df[df[LABEL_COL].isin([0, 1] + list(extra_neg))].reset_index(drop=True)
        raw_labels = df[LABEL_COL].to_numpy().astype(int)
        taluk_ids = df[TALUK_COL].astype(str).to_numpy()
        X = df[seq_cols].to_numpy(dtype=np.float32, copy=True).reshape(-1, N_YEARS, N_BANDS)
        X[~np.isfinite(X)] = np.nan

    taluks = np.array(sorted(np.unique(taluk_ids)))
    lab = np.array([int(raw_labels[taluk_ids == t][0] == 1) for t in taluks])
    cat = np.array([int(raw_labels[taluk_ids == t][0]) for t in taluks])
    codes = pd.Categorical(taluk_ids, categories=taluks).codes
    idx_by_taluk = {t: np.where(codes == i)[0] for i, t in enumerate(taluks)}
    return X, codes, taluks, lab, cat, idx_by_taluk

def load_geometry(gpkg_path, taluks):
    import sqlite3, struct
    con = sqlite3.connect(gpkg_path)
    cur = con.cursor()
    cur.execute("SELECT column_name, srs_id FROM gpkg_geometry_columns WHERE table_name=?", (GPKG_TABLE,))
    geom_col, srs = cur.fetchone()
    cur.execute(f'SELECT "{GPKG_KEY}", "{geom_col}" FROM "{GPKG_TABLE}"')
    boxes = {}
    for k, blob in cur.fetchall():
        if blob is None or blob[:2] != b"GP": continue
        flags = blob[3]
        if ((flags >> 1) & 7) == 0: continue
        minx, maxx, miny, maxy = struct.unpack("<dddd" if flags & 1 else ">dddd", blob[8:40])
        boxes[str(k)] = (minx, maxx, miny, maxy)
    con.close()

    B = np.array([boxes[t] for t in taluks], dtype=float)
    if srs in (4326, 4269, 4258):
        kx, ky = 111.32 * np.cos(np.radians(REF_LAT)), 110.57
        B = B * np.array([kx, kx, ky, ky])
    else:
        B = B / 1000.0
    dx = np.maximum(0, np.maximum(B[:, None, 0] - B[None, :, 1], B[None, :, 0] - B[:, None, 1]))
    dy = np.maximum(0, np.maximum(B[:, None, 2] - B[None, :, 3], B[None, :, 2] - B[:, None, 3]))
    return np.sqrt(dx ** 2 + dy ** 2)

def fit_norm(Xtr):
    flat = Xtr.reshape(-1, N_BANDS)
    mean = np.nan_to_num(np.nanmean(flat, axis=0))
    std = np.nanstd(flat, axis=0)
    std = np.where(~np.isfinite(std) | (std < 1e-6), 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)

def apply_norm(X, mean, std):
    miss = (~np.isfinite(X)).astype(np.float32)
    Z = np.clip(np.nan_to_num((X - mean) / std, nan=0.0), -5.0, 5.0)
    return np.concatenate([Z, miss], axis=2).astype(np.float32)

def make_weights(codes_tr, y_tr):
    s = pd.DataFrame({"t": codes_tr, "y": y_tr})
    n_patch = s.groupby("t")["t"].transform("size").to_numpy()
    n_tal_cls = s.groupby("y")["t"].transform("nunique").to_numpy()
    w = 1.0 / (n_patch * n_tal_cls)
    return (w * len(w) / w.sum()).astype(np.float32)

# ----------------------------------------------------------------------------- ARCHITECTURES
class BiGRU(nn.Module):
    """LayerNorm -> Linear -> Bidirectional GRU (hidden 16) -> MLP head."""
    def __init__(self, n_in=2 * N_BANDS, hidden=16, dropout=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(n_in)
        self.proj = nn.Sequential(nn.Linear(n_in, 32), nn.GELU())
        self.gru = nn.GRU(32, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        _, h = self.gru(self.proj(self.norm(x)))
        return self.head(torch.cat([h[0], h[1]], dim=1)).squeeze(-1)

class Conv1D(nn.Module):
    """Temporal 1D-CNN: Kernel=3 filters capturing 3-year moving trends & curvature."""
    def __init__(self, n_in=2 * N_BANDS, channels=32, dropout=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(n_in)
        self.conv1 = nn.Conv1d(n_in, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(channels, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        x = self.norm(x).transpose(1, 2)  # (B, Channels, Time)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        return self.head(x).squeeze(-1)

class TemporalTransformer(nn.Module):
    """Small Temporal Transformer: 2-layer Encoder with Multi-Head Self-Attention."""
    def __init__(self, n_in=2 * N_BANDS, d_model=32, nhead=4, num_layers=2, dropout=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(n_in)
        self.proj = nn.Linear(n_in, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, N_YEARS, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=64,
                                                   dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        h = self.proj(self.norm(x)) + self.pos_emb
        h = self.transformer(h)
        h = h.mean(dim=1)  # Temporal pooling
        return self.head(h).squeeze(-1)

def train_and_eval(model_class, Ztr, ytr, wtr, Zte, seed, epochs=EPOCHS):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model_class().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(TensorDataset(torch.from_numpy(Ztr), torch.from_numpy(ytr).float(), torch.from_numpy(wtr)),
                        batch_size=BATCH, shuffle=True)
    for ep in range(epochs):
        model.train()
        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            xb = xb.clone()
            xb[..., :N_BANDS] += NOISE * torch.randn_like(xb[..., :N_BANDS])
            loss = (F.binary_cross_entropy_with_logits(model(xb), yb, reduction="none") * wb).sum() / wb.sum()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(Zte), 1024):
            out.append(torch.sigmoid(model(torch.from_numpy(Zte[i:i + 1024]).to(device))).cpu().numpy())
    return np.mean(np.concatenate(out))

# ----------------------------------------------------------------------------- MAIN
def main():
    print("=" * 72)
    print("  STAGE 3: DEEP SEQUENCE LEARNING BENCHMARK (Bi-GRU & 1D-CNN)")
    print("=" * 72)
    print(f"Device: {device} | Buffer: {BUFFER_KM} km | Epochs: {EPOCHS} | Optimizer: AdamW")
    t0 = time.time()

    X, codes, taluks, lab, cat, idx_by_taluk = load_data(extra_neg=(-1,))
    D = load_geometry(GPKG_PATH, taluks)
    n = len(taluks)
    n_purged = (D <= BUFFER_KM).sum(1) - 1

    scores_gru = np.full(n, np.nan)
    scores_cnn = np.full(n, np.nan)
    scores_xformer = np.full(n, np.nan)

    print(f"\nRunning 54 LOTO Folds across Bi-GRU, 1D-CNN, and Small Transformer ...")
    for i in range(n):
        tr_t = np.where(D[i] > BUFFER_KM)[0]
        ytr = lab[codes[np.isin(codes, tr_t)]]
        if min((ytr == 0).sum(), (ytr == 1).sum()) < 2:
            continue

        tr_mask = np.isin(codes, tr_t)
        mean, std = fit_norm(X[tr_mask])
        Ztr = apply_norm(X[tr_mask], mean, std)
        Zte = apply_norm(X[idx_by_taluk[taluks[i]]], mean, std)
        ytr = lab[codes[tr_mask]]
        wtr = make_weights(codes[tr_mask], ytr)

        # 1. Train Bi-GRU
        scores_gru[i] = train_and_eval(BiGRU, Ztr, ytr, wtr, Zte, seed=SEED + i)
        # 2. Train Temporal 1D-CNN
        scores_cnn[i] = train_and_eval(Conv1D, Ztr, ytr, wtr, Zte, seed=SEED + 500 + i)
        # 3. Train Small Temporal Transformer
        scores_xformer[i] = train_and_eval(TemporalTransformer, Ztr, ytr, wtr, Zte, seed=SEED + 1000 + i)

        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"  --> Progress: [{i + 1:02d}/{n}] folds done (elapsed: {time.time() - t0:.1f}s)")

    print("\n" + "=" * 72)
    print(f"{'Deep Learning Model':<22} {'ROC-AUC':<9} {'Accuracy':<10} {'Balanced Acc'}")
    print("-" * 72)

    dl_models = [("DL_Bi_GRU", scores_gru), ("DL_1D_CNN", scores_cnn), ("DL_Transformer", scores_xformer)]
    for name, s_arr in dl_models:
        auc = roc_auc_score(lab, s_arr)
        pred = (s_arr >= 0.50).astype(int)
        acc = (pred == lab).mean() * 100
        bacc = balanced_accuracy_score(lab, pred) * 100
        print(f"{name:<22} {auc*100:5.1f}%    {acc:5.1f}%     {bacc:5.1f}%")

    print("=" * 72)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, "stage3_deep_learning_scores.csv")
    pd.DataFrame({"taluk": taluks, "label": lab, "n_purged": n_purged,
                  "DL_Bi_GRU": scores_gru, "DL_1D_CNN": scores_cnn,
                  "DL_Transformer": scores_xformer}).to_csv(out_csv, index=False)

    print(f"\nSaved Stage 3 results to -> {out_csv}")
    print(f"Total Stage 3 runtime: {time.time() - t0:.2f} seconds.")

if __name__ == "__main__":
    main()
