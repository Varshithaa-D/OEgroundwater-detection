import os; os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, confusion_matrix,
                              ConfusionMatrixDisplay)

OUT_DIR = "runs/dl"
RES_DIR = f"{OUT_DIR}/results"
os.makedirs(RES_DIR, exist_ok=True)

MODEL_NAMES = {"resnet": "ResNet-18", "vit": "ViT-Small", "grucnn": "CNN+GRU"}
COLORS      = {"resnet": "#e74c3c", "vit": "#2980b9", "grucnn": "#27ae60"}


def load_results():
    results = {}
    for f in glob.glob(f"{OUT_DIR}/*.json"):
        d = json.load(open(f))
        key = f"{d['model']}_{d['split']}"
        results[key] = d
    return results


def print_table(results):
    header = f"{'Model':<14} {'Split':<9} {'Accuracy':>9} {'F1-Macro':>9} {'AUC':>8} {'Precision':>10} {'Recall':>8}"
    print("\n" + "=" * 70)
    print("  COMPARISON TABLE — ALL MODELS & SPLITS")
    print("=" * 70)
    print(header)
    print("-" * 70)
    for key, d in sorted(results.items()):
        name = MODEL_NAMES.get(d["model"], d["model"])
        split = "Spatial Block (1.5km)" if d["split"] == "block" else "Grouped CV"
        print(f"{name:<14} {split:<12} "
              f"{d['accuracy']*100:>7.1f}%  "
              f"{d['f1_macro']:>8.3f}  "
              f"{d['auc']:>7.3f}  "
              f"{d['precision']:>9.3f}  "
              f"{d['recall']:>7.3f}")
    print("=" * 70)


def plot_roc_curves(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, split in zip(axes, ["block", "grouped"]):
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Chance (AUC=0.50)")
        for model in ["resnet", "vit", "grucnn"]:
            key = f"{model}_{split}"
            if key not in results: continue
            d = results[key]
            labs   = np.array(d["labels"])
            probs  = np.array(torch_sigmoid(d["logits"]))
            fpr, tpr, _ = roc_curve(labs, probs)
            roc_auc = auc(fpr, tpr)
            label = f"{MODEL_NAMES[model]} (AUC={roc_auc:.3f})"
            ax.plot(fpr, tpr, color=COLORS[model], lw=2, label=label)
        title = "Spatial Block Holdout (Strict)" if split == "block" else "Taluk-Grouped CV (Strict)"
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = f"{RES_DIR}/roc_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC curves → {path}")


def plot_confusion_matrices(results):
    models = ["resnet", "vit", "grucnn"]
    splits = ["block", "grouped"]
    fig, axes = plt.subplots(len(splits), len(models),
                             figsize=(13, 8))
    for j, split in enumerate(splits):
        for i, model in enumerate(models):
            key = f"{model}_{split}"
            ax  = axes[j][i]
            if key not in results:
                ax.axis("off"); continue
            d     = results[key]
            labs  = np.array(d["labels"]).astype(int)
            probs = np.array(torch_sigmoid(d["logits"]))
            preds = (probs > 0.5).astype(int)
            cm    = confusion_matrix(labs, preds)
            disp  = ConfusionMatrixDisplay(cm, display_labels=["Safe", "OE"])
            disp.plot(ax=ax, colorbar=False, cmap="Blues")
            split_label = "Block" if split == "block" else "Grouped"
            ax.set_title(f"{MODEL_NAMES[model]}\n({split_label})", fontsize=10)
    plt.suptitle("Confusion Matrices — Safe vs Over-Exploited Taluks",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = f"{RES_DIR}/confusion_matrices.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrices → {path}")


def plot_training_loss(results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, model in zip(axes, ["resnet", "vit", "grucnn"]):
        for split, ls in [("block", "-"), ("grouped", "--")]:
            key = f"{model}_{split}"
            if key not in results: continue
            d = results[key]
            if split == "block":
                hist = d.get("history", [])
                epochs = [h["epoch"] for h in hist]
                losses = [h["loss"]  for h in hist]
                label  = "Spatial Block"
            else:
                # average across folds
                hists  = d.get("histories", [[]])
                epochs = list(range(1, len(hists[0]) + 1))
                losses = [np.mean([h[i]["loss"] for h in hists]) for i in range(len(hists[0]))]
                label  = "Grouped CV (avg)"
            ax.plot(epochs, losses, ls=ls, color=COLORS[model], lw=2, label=label)
        ax.set_title(MODEL_NAMES[model], fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("BCE Loss")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.suptitle("Training Loss Curves", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = f"{RES_DIR}/training_loss.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved loss curves → {path}")


def bar_comparison(results):
    models = ["resnet", "vit", "grucnn"]
    splits = ["block", "grouped"]
    metrics = ["accuracy", "f1_macro", "auc"]
    metric_labels = ["Accuracy", "F1-Macro", "AUC"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(models))
    width = 0.35
    for ax, metric, mlabel in zip(axes, metrics, metric_labels):
        for j, (split, ls) in enumerate([("block", "Spatial Block"), ("grouped", "Strict Grouped CV")]):
            vals = []
            for model in models:
                key = f"{model}_{split}"
                vals.append(results[key][metric] if key in results else 0)
            bars = ax.bar(x + j * width, vals, width, label=ls,
                          color=["#e74c3c", "#2980b9", "#27ae60"][j*0:(j+1)*3])
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(mlabel, fontsize=11, fontweight="bold")
        ax.set_xticks(x + width/2)
        ax.set_xticklabels([MODEL_NAMES[m] for m in models], fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
    plt.suptitle("Model Comparison",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = f"{RES_DIR}/bar_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved bar comparison → {path}")


def torch_sigmoid(logits):
    a = np.array(logits)
    return 1 / (1 + np.exp(-np.clip(a, -500, 500)))


def save_csv(results):
    rows = []
    for key, d in sorted(results.items()):
        rows.append({
            "Model":     MODEL_NAMES.get(d["model"], d["model"]),
            "Split":     "Spatial Block" if d["split"] == "block" else "Grouped CV",
            "Accuracy":  f"{d['accuracy']*100:.1f}%",
            "F1-Macro":  f"{d['f1_macro']:.3f}",
            "Precision": f"{d['precision']:.3f}",
            "Recall":    f"{d['recall']:.3f}",
            "AUC":       f"{d['auc']:.3f}",
            "N_test":    d["n_test"],
        })
    import pandas as pd
    df = pd.DataFrame(rows)
    path = f"{RES_DIR}/results_table.csv"
    df.to_csv(path, index=False)
    print(f"Saved results table → {path}")
    return df


if __name__ == "__main__":
    results = load_results()
    if not results:
        print("No results found! Run dl_train.py first.")
    else:
        print_table(results)
        plot_roc_curves(results)
        plot_confusion_matrices(results)
        plot_training_loss(results)
        bar_comparison(results)
        df = save_csv(results)
        print(f"\n✅ All plots saved to {RES_DIR}/")
        print("\nFinal Results Summary:")
        print(df.to_string(index=False))
