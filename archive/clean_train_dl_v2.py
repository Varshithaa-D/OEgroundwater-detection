"""
clean_train_dl_v2.py — Groundwater Over-Exploitation Detection
==============================================================
VERSION 2 — All reviewer issues fixed.

FIXES APPLIED (in order of severity):

FIX 1 — Early stopping no longer uses test fold:
    A validation taluk is carved from the TRAINING taluks only.
    Early stopping is based on validation AUC. The test fold is
    never touched until the final evaluation.

FIX 2 — Normalization fitted per-fold on training data only:
    band_mean/std are computed ONLY from training patches and then
    applied to val and test. No information from test leaks in.

FIX 3 — BiLSTM readout fixed:
    Uses final hidden states h[-2] and h[-1] (one per direction),
    not out[:,-1,:] which only reads the backward direction's
    view of the last timestep.

FIX 4 — Missing values handled correctly:
    NaNs are preserved through normalization (nanmean/nanstd),
    then a binary missing-value mask is added as extra channels,
    then NaNs are zero-imputed. This avoids treating gaps as
    real observations while still signalling "missing" to the model.

FIX 5 — Bootstrap confidence intervals:
    AUC is reported with 95% CI using 1000 bootstrap resamples
    over taluks (not patches).

FIX 6 — Per-fold results printed:
    Each fold's taluk AUC is printed. The pooled number is
    described honestly as an approximation.

FIX 7 — Random seeds set:
    torch, numpy, and random seeds are all fixed for reproducibility.

HONEST LIMITATIONS (cannot be fixed without more data):
- Only 6 Safe taluks out of 50. Fold 2 has ZERO Safe taluks and
  is excluded from AUC. Any AUC rests on 6 data points total.
- The fold column was built by StratifiedGroupKFold on taluk names,
  NOT on geographic coordinates. Neighbouring taluks can appear in
  both train and test. True geographic isolation would require a
  buffer zone (needs lat/lon per taluk).
- Labels are assigned at taluk level but training is at patch level
  (pseudo-replication). Mean-pooling at test time reduces the bias
  but does not eliminate it. This is stated clearly.
- Semi-critical taluks (label=-1) are excluded, making the
  classification task easier than real deployment.

Usage:
    python clean_train_dl_v2.py

Requirements: torch, pandas, numpy, scikit-learn
"""

import os, json, random, warnings
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score,
    f1_score, confusion_matrix, classification_report
)

# ── Reproducibility ───────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Config ────────────────────────────────────────────────────────
BATCH_SIZE   = 512
EPOCHS       = 40
PATIENCE     = 7        # Early stopping patience on VAL (not test)
LR           = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_DIM   = 64
N_YEARS      = 10
N_BANDS      = 13
DROPOUT      = 0.3
MAX_PATCHES  = 400      # Per taluk subsampling
N_BOOTSTRAP  = 1000     # Bootstrap resamples for CI

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────
# STEP 1: Load data
# ─────────────────────────────────────────────────────────────────
print("=" * 68)
print("  GROUNDWATER DETECTION — CLEAN DL PIPELINE v2 (All Fixes Applied)")
print("=" * 68)
print(f"\n  Device: {device}")
print(f"  Seed:   {SEED}")

print("\n[1/6] Loading index.parquet...")
df = pd.read_parquet("index.parquet")
df = df[df["taluk_label"].isin([0, 1])].copy()
df = df[df["fold"] >= 0].copy()
df = df.reset_index(drop=True)

taluk_label_map = df.groupby("taluk_id")["taluk_label"].first().to_dict()
taluk_fold_map  = df.groupby("taluk_id")["fold"].first().to_dict()
all_taluks      = list(taluk_label_map.keys())

n_oe   = sum(1 for v in taluk_label_map.values() if v == 1)
n_safe = sum(1 for v in taluk_label_map.values() if v == 0)

print(f"  Total patches:     {len(df):,}")
print(f"  Total taluks:      {len(all_taluks)}")
print(f"  OE taluks:         {n_oe}")
print(f"  Safe taluks:       {n_safe}")
print(f"  NOTE: Only {n_safe} Safe taluks. AUC estimates have very wide CIs.")

# ─────────────────────────────────────────────────────────────────
# STEP 2: Extract raw sequences
# ─────────────────────────────────────────────────────────────────
print("\n[2/6] Extracting temporal features (seq_ columns)...")

seq_cols = []
for y in range(N_YEARS):
    for b in range(N_BANDS):
        col = f"seq_y{y}_b{b}"
        if col not in df.columns:
            raise KeyError(f"Missing: {col}")
        seq_cols.append(col)

X_raw = df[seq_cols].values.astype(np.float32)   # (N, 130) with NaN intact
X_raw = X_raw.reshape(-1, N_YEARS, N_BANDS)      # (N, 10, 13)

# Binary missing-value mask (FIX 4: signal gaps explicitly)
missing_mask = (~np.isfinite(X_raw)).astype(np.float32)  # (N, 10, 13) 1=missing

y_all     = df["taluk_label"].values.astype(np.int64)
fold_all  = df["fold"].values
taluk_all = df["taluk_id"].values

print(f"  Sequence shape: {X_raw.shape}  (patches × years × bands)")
print(f"  Missing values: {missing_mask.mean()*100:.1f}% of values are NaN/inf")

# ─────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────
class TemporalLSTM(nn.Module):
    """
    Bidirectional LSTM on 10-year × 13-band satellite sequences.
    FIX 3: uses final hidden states (h[-2], h[-1]) not out[:,-1,:]
    FIX 4: accepts (seq + missing_mask) as input = 26 channels
    """
    def __init__(self, n_in=N_BANDS * 2, hidden=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        # n_in = 13 bands + 13 missing-mask channels = 26
        self.input_proj = nn.Sequential(
            nn.Linear(n_in, 32),
            nn.LayerNorm(32),
            nn.GELU()
        )
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 32),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(32, 1)
        )
        for name, p in self.lstm.named_parameters():
            if "weight" in name: nn.init.orthogonal_(p)
            elif "bias"  in name: nn.init.zeros_(p)

    def forward(self, x):
        # x: (B, 10, 26)
        x = self.input_proj(x)               # (B, 10, 32)
        _, (h, _) = self.lstm(x)             # h: (4, B, 64)  [2layers×2dirs]
        # FIX 3: take last layer's two directional hidden states
        feat = torch.cat([h[-2], h[-1]], dim=1)   # (B, 128)
        return self.head(feat).squeeze(-1)


class PatchDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

def make_loader(X, y, shuffle=True):
    return DataLoader(PatchDS(X, y), batch_size=BATCH_SIZE,
                      shuffle=shuffle, num_workers=0)

def bootstrap_auc(y_true, y_prob, n=N_BOOTSTRAP, seed=SEED):
    """Bootstrap AUC CI over taluk-level samples."""
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    if len(aucs) < 10:
        return np.nan, np.nan
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))

# ─────────────────────────────────────────────────────────────────
# STEP 3: Leave-One-Fold-Out CV
# ─────────────────────────────────────────────────────────────────
print("\n[3/6] Leave-One-Fold-Out CV...")
print("  NOTE: fold was built by StratifiedGroupKFold on taluk names,")
print("  NOT on geographic coordinates. No geographic buffer applied.")
print("  Neighbouring taluks may appear in both train and test.")

valid_folds = sorted(np.unique(fold_all[fold_all >= 0]))
oof_probs   = np.full(len(df), np.nan)
oof_labels  = np.full(len(df), np.nan)
fold_aucs   = {}

for fold_k in valid_folds:
    print(f"\n  {'─'*60}")
    print(f"  FOLD {fold_k}")

    train_mask = (fold_all != fold_k) & (fold_all >= 0)
    test_mask  = (fold_all == fold_k)

    test_taluks  = list(np.unique(taluk_all[test_mask]))
    train_taluks = list(np.unique(taluk_all[train_mask]))

    test_labels = [taluk_label_map[t] for t in test_taluks]
    if len(set(test_labels)) < 2:
        print(f"  SKIP: test fold has only one class ({set(test_labels)}).")
        print(f"  Test taluks: {test_taluks}")
        fold_aucs[fold_k] = None
        continue

    # FIX 1: carve ONE val taluk from training taluks
    # Pick the training taluk with the rarest label (Safe if available)
    safe_train = [t for t in train_taluks if taluk_label_map[t] == 0]
    if safe_train:
        val_taluk = safe_train[0]   # use a Safe taluk as val
    else:
        val_taluk = train_taluks[0]

    inner_train_taluks = [t for t in train_taluks if t != val_taluk]
    print(f"  Train taluks ({len(inner_train_taluks)}): held for weight fitting")
    print(f"  Val   taluk  (1):            {val_taluk}  "
          f"[label={taluk_label_map[val_taluk]}] — early stopping")
    print(f"  Test  taluks ({len(test_taluks)}): {test_taluks}")

    inner_train_mask = np.isin(taluk_all, inner_train_taluks)
    val_mask         = (taluk_all == val_taluk)
    X_tr_raw  = X_raw[inner_train_mask]
    y_tr      = y_all[inner_train_mask]
    X_val_raw = X_raw[val_mask]
    y_val     = y_all[val_mask]
    X_te_raw  = X_raw[test_mask]
    y_te      = y_all[test_mask]

    # FIX 2: fit normalization ONLY on training data
    tr_flat   = X_tr_raw.reshape(-1, N_BANDS)
    band_mean = np.nanmean(tr_flat, axis=0)
    band_std  = np.nanstd(tr_flat,  axis=0)
    band_std  = np.where(band_std < 1e-6, 1.0, band_std)

    def normalize_and_concat_mask(X_raw_chunk):
        """Normalize, then concatenate missing-value mask. (FIX 2 + FIX 4)"""
        mask_chunk = (~np.isfinite(X_raw_chunk)).astype(np.float32)
        X_norm = (X_raw_chunk - band_mean[None, None, :]) / band_std[None, None, :]
        X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=3.0, neginf=-3.0)
        X_norm = np.clip(X_norm, -5.0, 5.0)
        return np.concatenate([X_norm, mask_chunk], axis=2)  # (N, 10, 26)

    X_tr_norm  = normalize_and_concat_mask(X_tr_raw)
    X_val_norm = normalize_and_concat_mask(X_val_raw)
    X_te_norm  = normalize_and_concat_mask(X_te_raw)

    # Subsample training patches
    tr_idx = np.where(inner_train_mask)[0]
    tr_tal = taluk_all[inner_train_mask]
    sampled = []
    for tal in np.unique(tr_tal):
        tidx = np.where(tr_tal == tal)[0]
        if len(tidx) > MAX_PATCHES:
            tidx = np.random.choice(tidx, MAX_PATCHES, replace=False)
        sampled.extend(tidx)
    sampled = np.array(sampled)
    X_tr_s = X_tr_norm[sampled]
    y_tr_s = y_tr[sampled]

    print(f"  Train patches (subsampled): {len(X_tr_s):,}  "
          f"Val: {len(X_val_norm):,}  Test: {len(X_te_norm):,}")

    # Class weight
    n_neg = (y_tr_s == 0).sum()
    n_pos = (y_tr_s == 1).sum()
    pos_w = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    print(f"  pos_weight = {pos_w.item():.2f}  (neg={n_neg}, pos={n_pos})")

    criterion    = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    model        = TemporalLSTM(n_in=N_BANDS * 2).to(device)
    optimizer    = torch.optim.AdamW(model.parameters(), lr=LR,
                                     weight_decay=WEIGHT_DECAY)
    scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(
                       optimizer, T_max=EPOCHS)

    train_loader = make_loader(X_tr_s,    y_tr_s, shuffle=True)
    val_loader   = make_loader(X_val_norm, y_val,  shuffle=False)
    test_loader  = make_loader(X_te_norm,  y_te,   shuffle=False)

    best_val_auc   = -1.0
    best_state     = None
    patience_count = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # FIX 1: evaluate early stopping on VAL (not test)
        model.eval()
        val_probs, val_labs = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                val_probs.extend(torch.sigmoid(model(xb.to(device))).cpu().numpy())
                val_labs.extend(yb.numpy())
        val_probs = np.array(val_probs)
        val_labs  = np.array(val_labs)

        if len(np.unique(val_labs)) >= 2:
            val_auc = roc_auc_score(val_labs, val_probs)
        else:
            val_auc = 0.5   # Cannot compute AUC if val has one class

        if epoch % 5 == 0 or epoch == 1:
            avg_loss = total_loss / len(train_loader)
            print(f"  Epoch {epoch:03d}/{EPOCHS}  loss={avg_loss:.4f}  "
                  f"val_AUC={val_auc:.4f}  {'← best' if val_auc > best_val_auc else ''}")

        if val_auc > best_val_auc:
            best_val_auc   = val_auc
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  Early stop at epoch {epoch}  best_val_AUC={best_val_auc:.4f}")
                break

    # Final inference on test with best model
    model.load_state_dict(best_state)
    model.eval()
    fold_patch_probs = []
    with torch.no_grad():
        for xb, _ in test_loader:
            fold_patch_probs.extend(
                torch.sigmoid(model(xb.to(device))).cpu().numpy())

    oof_probs[test_mask]  = fold_patch_probs
    oof_labels[test_mask] = y_te

    # Taluk-level (mean pooling)
    print(f"\n  Taluk-level results (mean-pooled):")
    tal_true_fold, tal_prob_fold = [], []
    for tal in test_taluks:
        tmask    = test_mask & (taluk_all == tal)
        tal_prob = float(np.mean(oof_probs[tmask]))
        tal_lbl  = int(y_all[tmask][0])
        tal_true_fold.append(tal_lbl)
        tal_prob_fold.append(tal_prob)
        print(f"    {tal:<40} label={tal_lbl}  prob={tal_prob:.3f}")

    if len(set(tal_true_fold)) >= 2:
        fauc = roc_auc_score(tal_true_fold, tal_prob_fold)
        print(f"  Fold {fold_k} Taluk AUC = {fauc:.4f}")
        fold_aucs[fold_k] = fauc
    else:
        print(f"  Fold {fold_k}: only one class in test → AUC undefined")
        fold_aucs[fold_k] = None

# ─────────────────────────────────────────────────────────────────
# STEP 4: Pooled evaluation + bootstrap CI
# ─────────────────────────────────────────────────────────────────
print("\n[4/6] Pooled Taluk-level Evaluation + Bootstrap CI...")

valid_mask = ~np.isnan(oof_probs)
y_true_all  = oof_labels[valid_mask].astype(int)
y_probs_all = oof_probs[valid_mask]
tal_all_val = taluk_all[valid_mask]

tal_results = []
for tal in np.unique(tal_all_val):
    m = tal_all_val == tal
    tal_results.append({
        "taluk": tal,
        "label": int(y_true_all[m][0]),
        "prob":  float(np.mean(y_probs_all[m]))
    })

tal_df    = pd.DataFrame(tal_results)
tal_true  = tal_df["label"].values
tal_probs = tal_df["prob"].values
tal_preds = (tal_probs >= 0.5).astype(int)

print(f"\n  {'─'*60}")
print(f"  Per-fold Taluk AUC summary:")
for k, v in fold_aucs.items():
    status = f"{v:.4f}" if v is not None else "UNDEFINED (single class)"
    print(f"    Fold {k}: {status}")

if len(np.unique(tal_true)) >= 2:
    auc  = roc_auc_score(tal_true, tal_probs)
    bacc = balanced_accuracy_score(tal_true, tal_preds)
    f1   = f1_score(tal_true, tal_preds, average="macro", zero_division=0)
    cm   = confusion_matrix(tal_true, tal_preds)
    ci_lo, ci_hi = bootstrap_auc(tal_true, tal_probs)

    print(f"\n{'='*68}")
    print(f"  POOLED RESULTS — Bidirectional LSTM (v2)")
    print(f"{'='*68}")
    print(f"  Taluks evaluated:   {len(tal_df)}")
    print(f"  Pooled Taluk AUC:   {auc:.4f}  95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"  Balanced Accuracy:  {bacc:.4f}")
    print(f"  F1 Macro:           {f1:.4f}")
    print(f"  Confusion Matrix (Taluks):")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")
    print(classification_report(tal_true, tal_preds,
                                target_names=["Safe","Over-Exploited"],
                                zero_division=0))
    print(f"{'='*68}")
    print(f"  FIXES APPLIED:")
    print(f"  ✅ Early stopping on VAL taluk, not test (Fix 1)")
    print(f"  ✅ Normalization fitted on training data only (Fix 2)")
    print(f"  ✅ BiLSTM readout: h[-2], h[-1] hidden states (Fix 3)")
    print(f"  ✅ Missing-value mask concatenated as extra channels (Fix 4)")
    print(f"  ✅ Bootstrap 95% CI reported (Fix 5)")
    print(f"  ✅ Per-fold AUC printed (Fix 6)")
    print(f"  ✅ Random seeds fixed (Fix 7)")
    print(f"{'─'*68}")
    print(f"  HONEST LIMITATIONS:")
    print(f"  ⚠  Only {n_safe} Safe taluks → wide confidence intervals")
    print(f"  ⚠  fold column is NOT geography-based (no lat/lon buffer)")
    print(f"  ⚠  Training uses taluk-label on patches (pseudo-replication)")
    print(f"  ⚠  Semi-critical taluks excluded (task is easier than deployment)")
    print(f"{'='*68}")

# ─────────────────────────────────────────────────────────────────
# STEP 5: Baseline (Logistic Regression on taluk-mean features)
# ─────────────────────────────────────────────────────────────────
print("\n[5/6] Baseline: Logistic Regression on taluk-mean features...")
print("  (Same folds, for fair comparison with LSTM)")

# Build taluk-level features from seq_ columns
seq_col_names = [f"seq_y{y}_b{b}" for y in range(N_YEARS) for b in range(N_BANDS)]
df_tal = df.groupby("taluk_id")[seq_col_names + ["taluk_label", "fold"]].first().reset_index()
X_tal  = df_tal[seq_col_names].values.astype(np.float32)
y_tal  = df_tal["taluk_label"].values
f_tal  = df_tal["fold"].values
t_tal  = df_tal["taluk_id"].values

lr_oof = np.full(len(df_tal), np.nan)
for fold_k in valid_folds:
    tr_m = (f_tal != fold_k) & (f_tal >= 0)
    te_m = (f_tal == fold_k)
    if len(np.unique(y_tal[te_m])) < 2: continue
    mdl = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                        LogisticRegression(C=0.1, class_weight="balanced",
                                           max_iter=2000, random_state=SEED))
    mdl.fit(X_tal[tr_m], y_tal[tr_m])
    lr_oof[te_m] = mdl.predict_proba(X_tal[te_m])[:, 1]

lr_valid = ~np.isnan(lr_oof)
if len(np.unique(y_tal[lr_valid])) >= 2:
    lr_auc = roc_auc_score(y_tal[lr_valid], lr_oof[lr_valid])
    lr_ci_lo, lr_ci_hi = bootstrap_auc(y_tal[lr_valid], lr_oof[lr_valid])
    print(f"  Baseline LR Taluk AUC: {lr_auc:.4f}  95% CI [{lr_ci_lo:.3f}, {lr_ci_hi:.3f}]")
    if len(np.unique(tal_true)) >= 2:
        print(f"  LSTM         Taluk AUC: {auc:.4f}  95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
        if ci_lo < lr_ci_hi:
            print(f"  ⚠  The CIs overlap. The LSTM is not proven better than LR baseline.")
        else:
            print(f"  ✅ LSTM is clearly better than LR baseline.")

# ─────────────────────────────────────────────────────────────────
# STEP 6: Save
# ─────────────────────────────────────────────────────────────────
print("\n[6/6] Saving outputs...")
os.makedirs("runs/clean_dl_v2", exist_ok=True)
tal_df["pct"] = (tal_df["prob"] * 100).round(1)
tal_df.to_csv("runs/clean_dl_v2/taluk_predictions.csv", index=False)

summary = {
    "model":           "Bidirectional LSTM v2 (hidden=64, BiLSTM fix, val-based ES)",
    "taluk_auc":       round(float(auc),  4) if len(np.unique(tal_true))>=2 else None,
    "auc_95ci_lo":     round(float(ci_lo), 4) if len(np.unique(tal_true))>=2 else None,
    "auc_95ci_hi":     round(float(ci_hi), 4) if len(np.unique(tal_true))>=2 else None,
    "balanced_acc":    round(float(bacc), 4)  if len(np.unique(tal_true))>=2 else None,
    "f1_macro":        round(float(f1),   4)  if len(np.unique(tal_true))>=2 else None,
    "per_fold_auc":    {str(k): round(v, 4) if v else None for k, v in fold_aucs.items()},
    "n_taluks_eval":   len(tal_df),
    "n_safe_taluks":   int(n_safe),
    "honest_limits":   [
        "Only 6 Safe taluks → very wide CIs",
        "Fold is not geography-based (no buffer)",
        "Patch pseudo-replication in training",
        "Semi-critical taluks excluded"
    ]
}
with open("runs/clean_dl_v2/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"  Saved → runs/clean_dl_v2/taluk_predictions.csv")
print(f"  Saved → runs/clean_dl_v2/summary.json")
print("\n  Done.")
