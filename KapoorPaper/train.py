"""
Training script for Skin Disease Multi-Classification
Based on: Kapoor et al. (2025), Section 3, Table 4

Exact training protocol from paper:
  - Optimizer : Adam  (Section 3)
  - Batch size: 32    (Section 3)
  - Epochs    : 20    (Section 3)
  - Loss      : CrossEntropy with class weights (Section 3.1)
  - Metrics   : Accuracy, Precision, Recall, F1-Score (Section 5)

Paper results (Table 4):
  EfficientNetB0 : acc=96.76%, prec=96.84%, recall=96.76%, f1=96.77%
  ResNet50       : acc=93.51%, prec=93.66%, recall=93.51%, f1=93.33%
  InceptionV3    : acc=93.51%, prec=94.09%, recall=93.51%, f1=93.53%
  VGG16          : acc=84.32%, prec=84.57%, recall=84.32%, f1=83.15%
"""

import os
import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)

from models import build_model, CLASS_NAMES, NUM_CLASSES, PAPER_RESULTS
from dataset import build_dataloaders


# ── Performance metrics (Section 5, Equations 1-4) ────────────────────────────

def compute_metrics(y_true, y_pred, class_names=CLASS_NAMES):
    """
    Compute the four metrics used in the paper (Section 5):
      Eq.1  Accuracy  = (TP+TN) / (TP+TN+FP+FN)
      Eq.2  Recall    = TP / (TP+FN)
      Eq.3  Precision = TP / (TP+FP)
      Eq.4  F1-Score  = 2 * (Precision × Recall) / (Precision + Recall)
    Uses 'weighted' averaging across 8 classes.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc  = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, average="weighted",
                           zero_division=0) * 100
    rec  = recall_score(y_true, y_pred, average="weighted",
                        zero_division=0) * 100
    f1   = f1_score(y_true, y_pred, average="weighted",
                    zero_division=0) * 100

    return {
        "accuracy":  round(acc, 2),
        "precision": round(prec, 2),
        "recall":    round(rec, 2),
        "f1":        round(f1, 2),
    }


def print_classification_report(y_true, y_pred, class_names=CLASS_NAMES):
    """Print per-class report matching paper's confusion matrices."""
    print("\n── Per-class Report ──────────────────────────────────────")
    print(classification_report(
        y_true, y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
    ))


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, is_inception):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # InceptionV3 returns (logits, aux_logits) in training mode
        if is_inception:
            outputs = model(imgs)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
        else:
            logits = model(imgs)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    metrics    = compute_metrics(all_labels, all_preds)
    return epoch_loss, metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, is_inception):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(imgs)
        if isinstance(logits, tuple):   # safety for inception
            logits = logits[0]

        loss = criterion(logits, labels)
        running_loss += loss.item() * imgs.size(0)

        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    metrics    = compute_metrics(all_labels, all_preds)
    return epoch_loss, metrics, all_labels, all_preds


# ── Main training function ────────────────────────────────────────────────────

def train(
    model_name,
    train_dir,
    test_dir,
    output_dir     = "checkpoints",
    num_epochs     = 20,          # Paper: 20 epochs
    batch_size     = 32,          # Paper: batch size 32
    lr             = 1e-3,        # Adam default (paper uses Adam optimizer)
    num_classes    = NUM_CLASSES,
    num_workers    = 4,
    freeze_base    = False,        # Paper: fine-tuning (not frozen)
    resume         = None,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_inception = (model_name == "inception_v3")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, class_weights = build_dataloaders(
        train_dir, test_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        model_name=model_name,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model, device = build_model(
        model_name,
        num_classes=num_classes,
        freeze_base=freeze_base,
        device=device,
    )

    # ── Loss with class weights (Section 3.1: "class weighting is used") ─────
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device)
    )

    # ── Optimizer: Adam (paper Section 3) ─────────────────────────────────────
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    start_epoch = 1
    best_acc    = 0.0
    history     = []

    # ── Resume ────────────────────────────────────────────────────────────────
    if resume and os.path.isfile(resume):
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_acc    = ckpt.get("best_acc", 0.0)
        print(f"[Train] Resumed from epoch {ckpt['epoch']}")

    ref = PAPER_RESULTS.get(model_name, {})
    print(f"\n{'='*65}")
    print(f" {model_name.upper()} | {num_epochs} epochs | batch={batch_size} | Adam lr={lr}")
    if ref:
        print(f" Paper target: acc={ref['accuracy']}%  f1={ref['f1']}%")
    print(f"{'='*65}\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, num_epochs + 1):
        t0 = time.time()

        tr_loss, tr_m = train_one_epoch(
            model, train_loader, criterion, optimizer, device, is_inception)
        va_loss, va_m, _, _ = evaluate(
            model, val_loader, criterion, device, is_inception)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{num_epochs} | {elapsed:.1f}s\n"
            f"  TRAIN  loss={tr_loss:.4f}  acc={tr_m['accuracy']:.2f}%  "
            f"prec={tr_m['precision']:.2f}%  rec={tr_m['recall']:.2f}%  "
            f"f1={tr_m['f1']:.2f}%\n"
            f"  VAL    loss={va_loss:.4f}  acc={va_m['accuracy']:.2f}%  "
            f"prec={va_m['precision']:.2f}%  rec={va_m['recall']:.2f}%  "
            f"f1={va_m['f1']:.2f}%"
        )

        row = {
            "epoch": epoch,
            "train_loss": tr_loss, **{f"train_{k}": v for k, v in tr_m.items()},
            "val_loss":   va_loss, **{f"val_{k}":   v for k, v in va_m.items()},
        }
        history.append(row)

        # Best checkpoint based on validation accuracy (not test)
        if va_m["accuracy"] > best_acc:
            best_acc = va_m["accuracy"]
            ckpt_path = os.path.join(output_dir, f"best_{model_name}.pth")
            torch.save({
                "epoch":       epoch,
                "model":       model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "best_acc":    best_acc,
                "val_metrics": va_m,
                "model_name":  model_name,
                "class_names": CLASS_NAMES,
            }, ckpt_path)
            print(f"  ✓ Best checkpoint saved → val_acc={best_acc:.2f}%")

        # Latest checkpoint
        torch.save({
            "epoch":      epoch,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "best_acc":   best_acc,
            "model_name": model_name,
        }, os.path.join(output_dir, f"last_{model_name}.pth"))

    # ── Final evaluation on test set (run only once, after training) ──────────
    print(f"\n{'='*65}")
    print(f" Final Test Results – {model_name}")
    print(f"{'='*65}")
    _, te_m, y_true, y_pred = evaluate(
        model, test_loader, criterion, device, is_inception)
    print(
        f"  TEST   acc={te_m['accuracy']:.2f}%  "
        f"prec={te_m['precision']:.2f}%  rec={te_m['recall']:.2f}%  "
        f"f1={te_m['f1']:.2f}%"
    )
    print_classification_report(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Save history & final metrics
    hist_path = os.path.join(output_dir, f"history_{model_name}.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    final_metrics_path = os.path.join(output_dir, f"metrics_{model_name}.json")
    with open(final_metrics_path, "w") as f:
        final = {
            "model": model_name,
            "best_val_accuracy": best_acc,
            "final_test_metrics": te_m,
            "paper_target": PAPER_RESULTS.get(model_name, {}),
            "confusion_matrix": cm.tolist(),
        }
        json.dump(final, f, indent=2)

    # Save misclassified image info
    test_samples = test_loader.dataset.samples  # list of (path, true_label)
    errors = [
        {
            "path":       path,
            "true_class": CLASS_NAMES[true_label],
            "pred_class": CLASS_NAMES[int(y_pred[i])],
            "true_idx":   true_label,
            "pred_idx":   int(y_pred[i]),
        }
        for i, (path, true_label) in enumerate(test_samples)
        if true_label != int(y_pred[i])
    ]
    errors_path = os.path.join(output_dir, f"errors_{model_name}.json")
    with open(errors_path, "w") as f:
        json.dump(errors, f, indent=2)
    print(f"[Train] Misclassified: {len(errors)}/{len(test_samples)} images "
          f"→ {errors_path}")

    print(f"\n[Train] Best val_acc={best_acc:.2f}%  "
          f"Final test_acc={te_m['accuracy']:.2f}%  "
          f"(paper target: {ref.get('accuracy','?')}%)")
    return history


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Skin Disease Classification – Kapoor et al. 2025"
    )
    parser.add_argument("--model",
        choices=["efficientnet_b0", "resnet50", "inception_v3", "vgg16"],
        required=True,
        help="CNN architecture to train")
    parser.add_argument("--train_dir",   required=True,
        help="Path to train/ folder of Kaggle dataset")
    parser.add_argument("--test_dir",    required=True,
        help="Path to test/ folder of Kaggle dataset")
    parser.add_argument("--output_dir",  default="checkpoints")
    parser.add_argument("--epochs",      type=int,   default=20,
        help="Number of epochs (paper: 20)")
    parser.add_argument("--batch_size",  type=int,   default=32,
        help="Batch size (paper: 32)")
    parser.add_argument("--lr",          type=float, default=1e-3,
        help="Learning rate for Adam optimizer")
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--freeze_base", action="store_true",
        help="Freeze backbone (default: fine-tune all layers)")
    parser.add_argument("--resume",      default=None,
        help="Path to checkpoint to resume training")
    args = parser.parse_args()

    train(
        model_name  = args.model,
        train_dir   = args.train_dir,
        test_dir    = args.test_dir,
        output_dir  = args.output_dir,
        num_epochs  = args.epochs,
        batch_size  = args.batch_size,
        lr          = args.lr,
        num_workers = args.num_workers,
        freeze_base = args.freeze_base,
        resume      = args.resume,
    )
