"""
Inference & model comparison for Skin Disease Multi-Classification
Replicates Table 4 of Kapoor et al. (2025):
  - Runs inference with a trained checkpoint on a single image or folder
  - Optionally trains all 4 models and prints comparison table
"""

import os
import json
import argparse
from pathlib import Path
import math

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")   # save to file without a display; remove for interactive
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from models import build_model, CLASS_NAMES, NUM_CLASSES, PAPER_RESULTS
from dataset import get_val_transforms, get_inception_transforms


# ── Visualisation helpers ─────────────────────────────────────────────────────

def plot_learning_curves(history_path, save_dir=None, show=False):
    """Plot train/val accuracy and loss curves from history JSON."""
    with open(history_path) as f:
        history = json.load(f)

    epochs      = [r["epoch"]          for r in history]
    train_acc   = [r["train_accuracy"] for r in history]
    val_acc     = [r["val_accuracy"]   for r in history]
    train_loss  = [r["train_loss"]     for r in history]
    val_loss    = [r["val_loss"]       for r in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Learning Curves – {Path(history_path).stem}", fontsize=13)

    ax1.plot(epochs, train_acc, label="Train", marker="o", markersize=3)
    ax1.plot(epochs, val_acc,   label="Val",   marker="o", markersize=3)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Accuracy"); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_loss, label="Train", marker="o", markersize=3)
    ax2.plot(epochs, val_loss,   label="Val",   marker="o", markersize=3)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.set_title("Loss"); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_dir, "learning_curves.png", show)


def plot_confusion_matrix_heatmap(metrics_path, save_dir=None, show=False):
    """Plot confusion matrix heatmap from metrics JSON."""
    with open(metrics_path) as f:
        data = json.load(f)

    cm    = np.array(data["confusion_matrix"])
    model = data.get("model", "")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True",      fontsize=11)
    ax.set_title(f"Confusion Matrix – {model}", fontsize=13)
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.yticks(rotation=0,  fontsize=8)
    plt.tight_layout()
    _save_or_show(fig, save_dir, "confusion_matrix.png", show)


def plot_error_images(errors_path, save_dir=None, show=False, max_images=24):
    """Display a grid of misclassified test images with true/pred labels."""
    with open(errors_path) as f:
        errors = json.load(f)

    if not errors:
        print("[Plot] No misclassified images – perfect score!")
        return

    errors = errors[:max_images]
    n      = len(errors)
    ncols  = min(6, n)
    nrows  = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.5, nrows * 2.8))
    axes = np.array(axes).reshape(-1)   # flatten for easy indexing
    fig.suptitle(
        f"Misclassified Images ({n} shown) – {Path(errors_path).stem}",
        fontsize=12,
    )

    for ax, err in zip(axes, errors):
        try:
            img = Image.open(err["path"]).convert("RGB")
        except Exception:
            img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        ax.imshow(img)
        ax.set_title(
            f"T: {err['true_class'].split('-')[-1]}\n"
            f"P: {err['pred_class'].split('-')[-1]}",
            fontsize=7, color="red",
        )
        ax.axis("off")

    for ax in axes[n:]:   # hide unused subplots
        ax.axis("off")

    plt.tight_layout()
    _save_or_show(fig, save_dir, "error_images.png", show)


def plot_predict_probs(result, save_path=None, show=False):
    """Bar chart of class probabilities for a single-image prediction."""
    classes = list(result["all_probs"].keys())
    probs   = list(result["all_probs"].values())
    colors  = ["steelblue"] * len(classes)
    pred    = result["predicted_class"]
    colors[classes.index(pred)] = "tomato"

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(classes, probs, color=colors)
    ax.set_xlabel("Probability (%)")
    ax.set_title(
        f"Prediction: {pred}  ({result['confidence']:.1f}%)\n"
        f"Image: {Path(result['image']).name}",
        fontsize=11,
    )
    ax.set_xlim(0, 100)
    for bar, prob in zip(bars, probs):
        if prob > 1:
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{prob:.1f}%", va="center", fontsize=8)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_all(checkpoint_dir, model_name, show=False):
    """
    Generate all plots for one model from the checkpoint directory:
      - learning_curves.png
      - confusion_matrix.png
      - error_images.png
    """
    save_dir = checkpoint_dir

    history_path = os.path.join(checkpoint_dir, f"history_{model_name}.json")
    metrics_path = os.path.join(checkpoint_dir, f"metrics_{model_name}.json")
    errors_path  = os.path.join(checkpoint_dir, f"errors_{model_name}.json")

    if os.path.isfile(history_path):
        plot_learning_curves(history_path, save_dir=save_dir, show=show)
    else:
        print(f"[Plot] Not found: {history_path}")

    if os.path.isfile(metrics_path):
        plot_confusion_matrix_heatmap(metrics_path, save_dir=save_dir, show=show)
    else:
        print(f"[Plot] Not found: {metrics_path}")

    if os.path.isfile(errors_path):
        plot_error_images(errors_path, save_dir=save_dir, show=show)
    else:
        print(f"[Plot] Not found: {errors_path}")


def _save_or_show(fig, save_dir, filename, show):
    if save_dir:
        path = os.path.join(save_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved → {path}")
    if show:
        plt.show()
    plt.close(fig)


# ── Single-image inference ────────────────────────────────────────────────────

@torch.no_grad()
def predict_image(model, img_path, model_name, device):
    """
    Predict the skin disease class for a single image.

    Returns dict with:
      predicted_class, confidence, all_probabilities
    """
    is_inception = (model_name == "inception_v3")
    transform    = (get_inception_transforms("test") if is_inception
                    else get_val_transforms(224))

    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    model.eval()
    logits = model(tensor)
    if isinstance(logits, tuple):
        logits = logits[0]

    probs     = F.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx  = int(np.argmax(probs))
    pred_cls  = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx]) * 100

    return {
        "image":           str(img_path),
        "predicted_class": pred_cls,
        "confidence":      round(confidence, 2),
        "all_probs": {
            cls: round(float(probs[i]) * 100, 2)
            for i, cls in enumerate(CLASS_NAMES)
        },
    }


# ── Load checkpoint & run inference ──────────────────────────────────────────

def load_and_infer(checkpoint_path, img_path, plot=False, show=False):
    """Load a saved checkpoint and run inference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model_name = ckpt.get("model_name", "efficientnet_b0")

    model, device = build_model(model_name, device=device)
    model.load_state_dict(ckpt["model"])

    print(f"[Infer] Model: {model_name} | "
          f"checkpoint acc: {ckpt.get('best_acc', '?')}%")

    result = predict_image(model, img_path, model_name, device)

    print(f"\n── Prediction ────────────────────────────────────")
    print(f"  Image     : {result['image']}")
    print(f"  Prediction: {result['predicted_class']}")
    print(f"  Confidence: {result['confidence']:.2f}%")
    print(f"\n── All class probabilities ───────────────────────")
    for cls, prob in sorted(result["all_probs"].items(),
                            key=lambda x: -x[1]):
        bar = "█" * int(prob / 5)
        print(f"  {cls:35s} {prob:6.2f}%  {bar}")

    if plot:
        save_path = Path(img_path).with_suffix("") \
                    .parent / f"{Path(img_path).stem}_probs.png"
        plot_predict_probs(result, save_path=str(save_path), show=show)

    return result


# ── Run all 4 models and print comparison table (Table 4) ─────────────────────

def compare_all_models(checkpoint_dir, test_dir, num_workers=4):
    """
    Evaluate all 4 models from saved checkpoints and print
    comparison table replicating Table 4 of the paper.
    """
    from dataset import SkinDiseaseDataset, get_val_transforms
    from train import evaluate
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_names = ["efficientnet_b0", "resnet50", "inception_v3", "vgg16"]
    results = {}

    for mn in model_names:
        ckpt_path = os.path.join(checkpoint_dir, f"best_{mn}.pth")
        if not os.path.isfile(ckpt_path):
            print(f"[Compare] Checkpoint not found: {ckpt_path} – skipped")
            continue

        ckpt = torch.load(ckpt_path, map_location=device)
        model, device = build_model(mn, device=device)
        model.load_state_dict(ckpt["model"])

        is_inception = (mn == "inception_v3")
        input_size   = 299 if is_inception else 224
        ds = SkinDiseaseDataset(
            test_dir,
            transform=get_val_transforms(input_size),
            mode="test",
        )
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, batch_size=32, shuffle=False,
                            num_workers=num_workers)
        criterion = nn.CrossEntropyLoss()
        _, metrics, _, _ = evaluate(model, loader, criterion,
                                    device, is_inception)
        results[mn] = metrics
        print(f"[Compare] {mn}: {metrics}")

    # Print Table 4
    print(f"\n{'='*75}")
    print(f" {'Parameter':12s} {'VGG-16':>12} {'InceptionV3':>12} "
          f"{'EfficientNetB0':>15} {'ResNet50':>12}")
    print(f"{'─'*75}")
    for metric in ["accuracy", "precision", "recall", "f1"]:
        row = f" {metric.capitalize():12s}"
        for mn in ["vgg16", "inception_v3", "efficientnet_b0", "resnet50"]:
            val = results.get(mn, {}).get(metric, "N/A")
            paper = PAPER_RESULTS.get(mn, {}).get(metric, "?")
            row += f" {str(val)+'%':>12}"
        print(row)
    print(f"{'─'*75}")
    print("  Paper Table 4 reference:")
    for metric in ["accuracy", "precision", "recall", "f1"]:
        row = f"  (paper) {metric:10s}"
        for mn in ["vgg16", "inception_v3", "efficientnet_b0", "resnet50"]:
            v = PAPER_RESULTS.get(mn, {}).get(metric, "?")
            row += f" {str(v)+'%':>12}"
        print(row)
    print(f"{'='*75}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference & comparison – Kapoor et al. 2025"
    )
    sub = parser.add_subparsers(dest="command")

    # predict sub-command
    pred_p = sub.add_parser("predict", help="Predict a single image")
    pred_p.add_argument("--checkpoint", required=True)
    pred_p.add_argument("--image",      required=True)
    pred_p.add_argument("--plot", action="store_true",
                        help="Save probability bar chart as PNG")
    pred_p.add_argument("--show", action="store_true",
                        help="Display plot interactively (requires display)")

    # compare sub-command
    cmp_p = sub.add_parser("compare",
        help="Compare all 4 models (requires all checkpoints)")
    cmp_p.add_argument("--checkpoint_dir", required=True)
    cmp_p.add_argument("--test_dir",       required=True)
    cmp_p.add_argument("--num_workers",    type=int, default=4)

    # plot sub-command  ── generate all visualisations for one model
    plt_p = sub.add_parser("plot",
        help="Generate all plots from training output JSONs")
    plt_p.add_argument("--checkpoint_dir", required=True,
                       help="Directory containing history/metrics/errors JSONs")
    plt_p.add_argument("--model", required=True,
                       choices=["efficientnet_b0", "resnet50",
                                "inception_v3", "vgg16"])
    plt_p.add_argument("--show", action="store_true",
                       help="Display plots interactively (requires display)")

    args = parser.parse_args()

    if args.command == "predict":
        load_and_infer(args.checkpoint, args.image,
                       plot=args.plot, show=args.show)

    elif args.command == "compare":
        compare_all_models(
            args.checkpoint_dir,
            args.test_dir,
            args.num_workers,
        )

    elif args.command == "plot":
        plot_all(args.checkpoint_dir, args.model, show=args.show)

    else:
        parser.print_help()
