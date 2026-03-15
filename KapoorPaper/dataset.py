"""
Dataset & DataLoader for Skin Disease Multi-Classification
Based on: Kapoor et al. (2025), Section 3.1 & 3.3

Dataset: Kaggle Skin-Disease-Dataset
  URL: https://www.kaggle.com/datasets/subirbiswas19/skin-disease-dataset
  Total: 1,159 images across 8 classes
  Split: 80% train (925) / 20% test (234)  [Table 2]

Expected folder structure (download from Kaggle and unzip):
  skin-disease-dataset/
  ├── train/
  │   ├── BA- cellulitis/
  │   ├── BA-impetigo/
  │   ├── FU-athlete-foot/
  │   ├── FU-nail-fungus/
  │   ├── FU-ringworm/
  │   ├── PA-cutaneous-larva-migrans/
  │   ├── VI-chickenpox/
  │   └── VI-shingles/
  └── test/
      └── (same 8 folders)

Pre-processing (Section 3.1):
  - Resize to 224×224 pixels
  - Model-specific normalization (ImageNet stats)
  - Stratified split (already done by Kaggle dataset)
  - Class weighting for imbalance
"""

import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from models import CLASS_NAMES, NUM_CLASSES


# ── Transforms (Section 3.1) ──────────────────────────────────────────────────
# Paper: "images are subsequently scaled to a fixed input size of 224 x 224"
# "normalization using model-specific preprocessing routines"

# ImageNet mean/std (used by all four pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms(input_size=224):
    """
    Training transforms with augmentation.
    Paper: "standardized preprocessing and augmentation" (Conclusion).
    """
    return transforms.Compose([
        transforms.Resize((input_size + 32, input_size + 32)),
        transforms.RandomCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(input_size=224):
    """Validation/test transforms (no augmentation, deterministic)."""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inception_transforms(split="train"):
    """InceptionV3 needs 299×299 input."""
    if split == "train":
        return get_train_transforms(input_size=299)
    return get_val_transforms(input_size=299)


# ── Dataset ───────────────────────────────────────────────────────────────────

class SkinDiseaseDataset(Dataset):
    """
    Skin disease dataset matching Kaggle folder structure.

    Folder names on Kaggle may have spaces/capitalisation variations.
    This loader maps them to canonical CLASS_NAMES automatically.
    """

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif"}

    def __init__(self, root_dir, transform=None, mode="train"):
        """
        Args:
            root_dir  : path to train/ or test/ directory
            transform : torchvision transform pipeline
            mode      : 'train' or 'test'/'val'
        """
        self.root_dir  = Path(root_dir)
        self.transform = transform
        self.mode      = mode
        self.samples   = []   # (img_path, label_idx)
        self.class_to_idx = {}
        self._load_samples()

    def _normalise_class_name(self, folder_name):
        """
        Map Kaggle folder names (which vary in capitalisation/spacing)
        to canonical CLASS_NAMES indices.
        Returns (canonical_name, index) or (None, None) if unrecognised.
        """
        # Strip and lowercase for fuzzy matching
        fn = folder_name.strip().lower().replace(" ", "-").replace("_", "-")
        for idx, cls in enumerate(CLASS_NAMES):
            if cls.lower() == fn:
                return cls, idx
        # Partial match fallback
        for idx, cls in enumerate(CLASS_NAMES):
            if cls.lower() in fn or fn in cls.lower():
                return cls, idx
        return None, None

    def _load_samples(self):
        found_classes = {}
        for cls_dir in sorted(self.root_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            cls_name, cls_idx = self._normalise_class_name(cls_dir.name)
            if cls_name is None:
                print(f"[Dataset] Warning: unrecognised folder '{cls_dir.name}' – skipped")
                continue
            found_classes[cls_name] = cls_idx
            self.class_to_idx[cls_name] = cls_idx
            for ext in self.IMG_EXTS:
                for p in cls_dir.glob(f"*{ext}"):
                    self.samples.append((str(p), cls_idx))
                for p in cls_dir.glob(f"*{ext.upper()}"):
                    self.samples.append((str(p), cls_idx))

        # Class distribution
        counts = {}
        for _, idx in self.samples:
            counts[idx] = counts.get(idx, 0) + 1
        print(f"[Dataset] mode={self.mode}  total={len(self.samples)}")
        for cls, idx in sorted(self.class_to_idx.items(), key=lambda x: x[1]):
            print(f"  {idx}: {cls:35s} {counts.get(idx, 0)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[Dataset] Cannot open {img_path}: {e}")
            img = Image.fromarray(
                np.zeros((224, 224, 3), dtype=np.uint8)
            )
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Subset wrapper with independent transform ─────────────────────────────────

class _SubsetWithTransform(Dataset):
    """Wraps a list of (path, label) samples with its own transform pipeline."""

    def __init__(self, samples, transform):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[Dataset] Cannot open {img_path}: {e}")
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Class weights for imbalanced data (Section 3.1) ──────────────────────────

def compute_class_weights(dataset, num_classes=NUM_CLASSES, device=None):
    """
    Paper: "class weighting is used" to overcome data imbalance.
    Returns inverse-frequency weights as a tensor.
    """
    counts = torch.zeros(num_classes)
    for _, label in dataset.samples:
        counts[label] += 1
    # Inverse frequency, normalised
    weights = counts.sum() / (num_classes * counts.clamp(min=1))
    if device:
        weights = weights.to(device)
    return weights


# ── DataLoader factory ────────────────────────────────────────────────────────

def build_dataloaders(
    train_dir,
    test_dir,
    batch_size=32,        # Paper: batch size of 32
    num_workers=4,
    model_name="efficientnet_b0",
    pin_memory=True,
    val_split=0.2,        # fraction of train data used for validation
):
    """
    Build train, val, and test DataLoaders.

    train data is stratified-split into train (1-val_split) / val (val_split).
    Test data is kept separate and used only for final evaluation.

    Args:
        train_dir  : path to train/ folder
        test_dir   : path to test/ folder
        batch_size : 32 (paper default)
        model_name : used to pick correct input size (299 for InceptionV3)
        val_split  : fraction of train images reserved for validation (default 0.2)

    Returns:
        train_loader, val_loader, test_loader, class_weights
    """
    # InceptionV3 needs 299×299 (Section 4.3)
    is_inception = (model_name == "inception_v3")
    input_size   = 299 if is_inception else 224

    # Load full train set without transforms to get sample list
    full_train_ds = SkinDiseaseDataset(train_dir, transform=None, mode="train")

    # Stratified split so class distribution is preserved in both subsets
    labels = [label for _, label in full_train_ds.samples]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
    train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))

    train_samples = [full_train_ds.samples[i] for i in train_idx]
    val_samples   = [full_train_ds.samples[i] for i in val_idx]

    train_ds = _SubsetWithTransform(train_samples, get_train_transforms(input_size))
    val_ds   = _SubsetWithTransform(val_samples,   get_val_transforms(input_size))
    test_ds  = SkinDiseaseDataset(
        test_dir,
        transform=get_val_transforms(input_size),
        mode="test",
    )

    # Class weights computed from the full train set (before split)
    class_weights = compute_class_weights(full_train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(f"\n[DataLoader] model={model_name}  input={input_size}px  "
          f"batch={batch_size}")
    print(f"[DataLoader] train={len(train_ds)}  val={len(val_ds)}  "
          f"test={len(test_ds)}")
    return train_loader, val_loader, test_loader, class_weights


if __name__ == "__main__":
    # Smoke test with dummy data
    import tempfile, shutil
    tmp = Path(tempfile.mkdtemp())
    for split in ["train", "test"]:
        for cls in CLASS_NAMES:
            d = tmp / split / cls
            d.mkdir(parents=True)
            for i in range(4):
                img = Image.fromarray(
                    np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                )
                img.save(str(d / f"img_{i}.jpg"))

    tl, vl, tel, cw = build_dataloaders(
        str(tmp / "train"),
        str(tmp / "test"),
        batch_size=4,
        num_workers=0,
    )
    imgs, labels = next(iter(tl))
    print(f"\n[Smoke test] batch: {imgs.shape}  labels: {labels}")
    print(f"[Smoke test] class weights: {cw.numpy().round(3)}")
    print(f"[Smoke test] val batches={len(vl)}  test batches={len(tel)}")
    shutil.rmtree(tmp)
