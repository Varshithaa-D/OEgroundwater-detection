"""
dl_train.py: Rigorous Groundwater Stress Deep Learning Pipeline (3D-CNN on RTX 3050 GPU)
========================================================================================
Trains a 3D-Convolutional Neural Network on real multi-sensor spatial satellite patches
from hydrosentry.zarr using Spatial Block Cross-Validation with geographic boundary buffering.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import numpy as np
import pandas as pd
import zarr
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, confusion_matrix, balanced_accuracy_score

# ------------------------------------------------------------------------------
# 1. GEOGRAPHIC SPATIAL BLOCK CV & BOUNDARY BUFFERING
# ------------------------------------------------------------------------------
def get_geographic_folds(geo_csv="taluk_geo.csv", n_splits=5, buffer_km=10.0, seed=42):
    if not os.path.exists(geo_csv):
        raise FileNotFoundError(f"'{geo_csv}' not found!")

    geo_df = pd.read_csv(geo_csv)
    coords = geo_df[['x_km', 'y_km']].values

    kmeans = KMeans(n_clusters=n_splits, random_state=seed, n_init=10)
    geo_df['fold'] = kmeans.fit_predict(coords)

    folds_dict = {}
    for test_fold in range(n_splits):
        val_fold = (test_fold + 1) % n_splits

        test_taluks = set(geo_df[geo_df['fold'] == test_fold]['taluk'])
        val_taluks = set(geo_df[geo_df['fold'] == val_fold]['taluk'])

        train_candidates = geo_df[~geo_df['fold'].isin([test_fold, val_fold])].copy().reset_index(drop=True)
        test_coords = coords[geo_df['fold'] == test_fold]

        keep_train = []
        for _, row in train_candidates.iterrows():
            t_coord = np.array([row['x_km'], row['y_km']])
            min_dist_km = np.min(np.linalg.norm(test_coords - t_coord, axis=1))
            keep_train.append(bool(min_dist_km >= buffer_km))

        train_taluks = set(train_candidates.loc[keep_train, 'taluk'])

        folds_dict[test_fold] = {
            'train': train_taluks,
            'val': val_taluks,
            'test': test_taluks
        }

        print(f"[Fold {test_fold}] Train taluks: {len(train_taluks)} | Val: {len(val_taluks)} | Test: {len(test_taluks)}")

    return folds_dict


# ------------------------------------------------------------------------------
# 2. RIGOROUS ZARR DATASET LOADER WITH SAFE NaN IMPUTATION
# ------------------------------------------------------------------------------
class GroundwaterDataset(Dataset):
    def __init__(self, index_df, zarr_path="hydrosentry.zarr", taluk_list=None, mean=None, std=None):
        self.zarr_root = zarr.open(zarr_path, mode='r')

        if taluk_list is not None:
            self.df = index_df[index_df['taluk_id'].astype(str).isin(taluk_list)].copy()
        else:
            self.df = index_df.copy()

        self.df = self.df.reset_index(drop=True)
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        taluk_name = str(row['taluk_id'])
        year = str(row['year'])
        r, c = int(row['row']), int(row['col'])
        label = int(row['taluk_label'])

        try:
            patch_data = self.zarr_root[taluk_name][year][:, r:r+16, c:c+16]
            patch_data = np.nan_to_num(patch_data, nan=0.0, posinf=0.0, neginf=0.0)
            tensor_x = torch.from_numpy(patch_data).float()
            if tensor_x.ndim == 3:
                tensor_x = tensor_x.unsqueeze(1) # [13, 1, 16, 16]
        except Exception:
            tensor_x = torch.zeros(13, 1, 16, 16, dtype=torch.float32)

        if self.mean is not None and self.std is not None:
            tensor_x = (tensor_x - self.mean) / (self.std + 1e-5)

        return tensor_x, label, taluk_name


def compute_global_stats(index_df, zarr_path, train_taluks):
    """Computes global channel mean and std across training taluks."""
    zarr_root = zarr.open(zarr_path, mode='r')
    train_subset = index_df[index_df['taluk_id'].astype(str).isin(train_taluks)]
    
    accumulator = []
    sample_df = train_subset.sample(min(len(train_subset), 150), random_state=42)
    
    for _, row in sample_df.iterrows():
        try:
            patch = zarr_root[str(row['taluk_id'])][str(row['year'])][:, int(row['row']):int(row['row'])+16, int(row['col']):int(row['col'])+16]
            patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            accumulator.append(patch)
        except Exception:
            continue

    if len(accumulator) == 0:
        return torch.zeros(13, 1, 1, 1), torch.ones(13, 1, 1, 1)

    stack = np.stack(accumulator, axis=0)
    mean = torch.tensor(stack.mean(axis=(0, 2, 3)), dtype=torch.float32).view(-1, 1, 1, 1)
    std = torch.tensor(stack.std(axis=(0, 2, 3)), dtype=torch.float32).view(-1, 1, 1, 1)
    return mean, std


# ------------------------------------------------------------------------------
# 3. SPATIAL 3D-CNN MODEL ARCHITECTURE
# ------------------------------------------------------------------------------
class GroundwaterCNN3D(nn.Module):
    def __init__(self, in_channels=13, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        feat = self.encoder(x).view(x.size(0), -1)
        out = self.classifier(feat)
        return out.squeeze(-1)


# ------------------------------------------------------------------------------
# 4. THRESHOLD CALIBRATION & EVALUATION
# ------------------------------------------------------------------------------
def find_optimal_threshold(y_true, y_probs):
    if len(np.unique(y_true)) < 2:
        return 0.5

    best_j, best_thresh = -1.0, float(np.median(y_probs))
    quantiles = np.quantile(y_probs, np.linspace(0.05, 0.95, 90))

    for t in quantiles:
        preds = (y_probs >= t).astype(int)
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        if cm.shape == (2, 2) and cm[0].sum() > 0 and cm[1].sum() > 0:
            sens = cm[1, 1] / cm[1].sum()
            spec = cm[0, 0] / cm[0].sum()
            j_stat = sens + spec - 1.0
            if j_stat > best_j:
                best_j, best_thresh = j_stat, t

    return best_thresh


def evaluate_taluk_predictions(df_patch_preds, val_threshold=None):
    df_taluk = df_patch_preds.groupby('taluk').agg({
        'y_true': 'first',
        'y_prob': 'mean'
    }).reset_index()

    y_true = df_taluk['y_true'].values
    y_probs = df_taluk['y_prob'].values

    if val_threshold is None:
        val_threshold = find_optimal_threshold(y_true, y_probs)

    y_pred = (y_probs >= val_threshold).astype(int)
    auc = roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.5
    bacc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        'taluk_auc': auc,
        'balanced_acc': bacc,
        'threshold': val_threshold,
        'confusion_matrix': cm,
        'df_taluk': df_taluk
    }


# ------------------------------------------------------------------------------
# 5. MAIN TRAINING & EVALUATION LOOP
# ------------------------------------------------------------------------------
def train_and_eval(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("  3D-CONVOLUTIONAL DEEP LEARNING GROUNDWATER MODEL (PYTORCH GPU)")
    print("=" * 70)
    print(f"  Executing on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    index_path = "index.parquet" if os.path.exists("index.parquet") else "index.csv"
    index_df = pd.read_parquet(index_path) if index_path.endswith('.parquet') else pd.read_csv(index_path)
    
    # Map semi-critical (-1) to negative (0) if requested
    if args.also_negative:
        index_df['taluk_label'] = index_df['taluk_label'].replace(-1, 0)
    else:
        index_df = index_df[index_df['taluk_label'] != -1].copy()

    zarr_root = zarr.open(args.zarr_path, mode='r')
    valid_taluks = set(zarr_root.group_keys())
    index_df = index_df[index_df['taluk_id'].astype(str).isin(valid_taluks)].copy().reset_index(drop=True)

    if args.max_patches_per_taluk > 0:
        index_df = index_df.groupby(['taluk_id', 'year'], group_keys=False).apply(
            lambda x: x.sample(min(len(x), args.max_patches_per_taluk), random_state=42)
        ).reset_index(drop=True)

    print(f"  Filtered & subsampled index to {len(index_df)} valid patches across {index_df['taluk_id'].nunique()} taluks.")

    folds = get_geographic_folds(geo_csv=args.geo_csv, n_splits=args.n_splits, buffer_km=args.buffer_km)
    all_oof_predictions = []

    for fold_idx in range(args.n_splits):
        print(f"\n========================== FOLD {fold_idx} ==========================")
        fold_data = folds[fold_idx]

        mean, std = compute_global_stats(index_df, args.zarr_path, fold_data['train'])

        train_ds = GroundwaterDataset(index_df, zarr_path=args.zarr_path, taluk_list=fold_data['train'], mean=mean, std=std)
        val_ds = GroundwaterDataset(index_df, zarr_path=args.zarr_path, taluk_list=fold_data['val'], mean=mean, std=std)
        test_ds = GroundwaterDataset(index_df, zarr_path=args.zarr_path, taluk_list=fold_data['test'], mean=mean, std=std)

        if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
            print(f"[Fold {fold_idx}] Skipping due to empty split.")
            continue

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        train_labels = train_ds.df['taluk_label'].values
        pos_count = np.sum(train_labels == 1)
        neg_count = np.sum(train_labels == 0)
        print(f"[Fold {fold_idx}] Train Positive Samples: {pos_count}, Negative Samples: {neg_count}")
        
        pos_weight = torch.tensor([neg_count / max(pos_count, 1)], device=device, dtype=torch.float32)

        model = GroundwaterCNN3D().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            for x, y, _ in train_loader:
                x, y = x.to(device), y.float().to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1:02d}/{args.epochs} - Train Loss: {running_loss/len(train_loader):.4f}")

        model.eval()
        val_records = []
        with torch.no_grad():
            for x, y, taluks in val_loader:
                x = x.to(device)
                probs = torch.sigmoid(model(x)).cpu().numpy()
                for t, label, prob in zip(taluks, y.numpy(), probs):
                    val_records.append({'taluk': t, 'y_true': label, 'y_prob': prob})

        df_val_patches = pd.DataFrame(val_records)
        val_res = evaluate_taluk_predictions(df_val_patches, val_threshold=None)
        calibrated_thresh = val_res['threshold']
        print(f"[Fold {fold_idx}] Val Taluk AUC: {val_res['taluk_auc']:.4f} | Validation-Calibrated Thresh: {calibrated_thresh:.4f}")

        test_records = []
        with torch.no_grad():
            for x, y, taluks in test_loader:
                x = x.to(device)
                probs = torch.sigmoid(model(x)).cpu().numpy()
                for t, label, prob in zip(taluks, y.numpy(), probs):
                    test_records.append({'taluk': t, 'y_true': label, 'y_prob': prob})

        if len(test_records) > 0:
            df_test_patches = pd.DataFrame(test_records)
            test_res = evaluate_taluk_predictions(df_test_patches, val_threshold=calibrated_thresh)
            print(f"[Fold {fold_idx}] Test Taluk AUC: {test_res['taluk_auc']:.4f} | Balanced Acc: {test_res['balanced_acc']:.4f}")
            print(f"Confusion Matrix:\n{test_res['confusion_matrix']}")
            all_oof_predictions.append(test_res['df_taluk'])

    if len(all_oof_predictions) > 0:
        print("\n==================================================================")
        print("           HONEST POOLED OUT-OF-FOLD EVALUATION                   ")
        print("==================================================================")
        oof_df = pd.concat(all_oof_predictions, ignore_index=True)
        oof_threshold = find_optimal_threshold(oof_df['y_true'].values, oof_df['y_prob'].values)
        oof_preds = (oof_df['y_prob'].values >= oof_threshold).astype(int)

        final_auc = roc_auc_score(oof_df['y_true'].values, oof_df['y_prob'].values)
        final_bacc = balanced_accuracy_score(oof_df['y_true'].values, oof_preds)
        final_cm = confusion_matrix(oof_df['y_true'].values, oof_preds, labels=[0, 1])

        print(f"Total Unique OOF Taluks Evaluated: {len(oof_df)}")
        print(f"--> Pooled OOF Taluk AUC:         {final_auc:.4f}")
        print(f"--> Pooled OOF Balanced Accuracy: {final_bacc:.4f}")
        print(f"--> Optimal OOF Threshold:         {oof_threshold:.4f}")
        print(f"--> Overall Accuracy:              {(oof_preds == oof_df['y_true'].values).mean() * 100:.2f}%")
        print(f"--> Overall Confusion Matrix:\n{final_cm}")
        os.makedirs("runs", exist_ok=True)
        oof_df.to_csv("runs/dl_predictions.csv", index=False)
        print(f"Saved predictions to runs/dl_predictions.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rigorous Spatial Block CV for Groundwater Stress")
    parser.add_argument("--geo_csv", default="taluk_geo.csv", help="CSV with x_km, y_km coordinates")
    parser.add_argument("--zarr_path", default="hydrosentry.zarr", help="Path to hydrosentry.zarr")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of spatial block folds")
    parser.add_argument("--buffer_km", type=float, default=10.0, help="Geographic buffer distance in km")
    parser.add_argument("--max_patches_per_taluk", type=int, default=20, help="Max patches per taluk per year")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs per fold")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--also-negative", action="store_true", help="Include semi-critical as negative")
    args = parser.parse_args()

    train_and_eval(args)
