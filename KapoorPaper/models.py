"""
Transfer Learning Models for Skin Disease Multi-Classification
Based on: Kapoor et al., "CNN-based Multi-Classification of Skin Disease
          with Fine-Tuned ResNet50 and VGG16"
          The Open Bioinformatics Journal, 2025, Vol.18

Four models as described in Sections 4, 4.1, 4.2, 4.3:
  - ResNet50   : GlobalAvgPool → Dense(512, ReLU) → Dense(8, Softmax)
  - VGG16      : GlobalAvgPool → Dense(256, ReLU) → Dropout(0.5) → Dense(8, Softmax)
  - EfficientNetB0: GlobalAvgPool → Dropout(0.1) → Dropout(0.3) → Dense(8, Softmax)
                   (all layers unfrozen for end-to-end fine-tuning)
  - InceptionV3: GlobalAvgPool → Dropout(0.1) → Dense(128, ReLU) → Dropout(0.3) → Dense(8, Softmax)

All base models: pretrained on ImageNet, top layers removed.
"""

import torch
import torch.nn as nn
import torchvision.models as models


# ── 8 disease classes (Table 2 in the paper) ─────────────────────────────────
CLASS_NAMES = [
    "BA-cellulitis",  # Bacterial
    "BA-impetigo",  # Bacterial
    "FU-athlete-foot",  # Fungal
    "FU-nail-fungus",  # Fungal
    "FU-ringworm",  # Fungal
    "PA-cutaneous-larva-migrans",  # Parasitic
    "VI-chickenpox",  # Viral
    "VI-shingles",  # Viral
]
NUM_CLASSES = len(CLASS_NAMES)  # 8


# ── ResNet50 (Section 4, Fig.5) ───────────────────────────────────────────────
# GlobalAvgPool → Dense(512, ReLU) → Dense(8, Softmax)


def build_resnet50(num_classes=NUM_CLASSES, freeze_base=False):
    """
    Proposed ResNet50 architecture (Fig.5):
      ResNet50 base (ImageNet weights, top removed)
      → GlobalAveragePooling2D
      → Dense(512, activation=ReLU)
      → Dense(num_classes, activation=Softmax)
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Optionally freeze base layers (feature extraction mode)
    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the original FC head with the paper's classification head
    in_features = model.fc.in_features  # 2048
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Linear(512, num_classes),
        # Note: Softmax is implicit in CrossEntropyLoss during training
        # Use softmax explicitly only at inference
    )
    return model


# ── VGG16 (Section 4.1, Fig.6) ───────────────────────────────────────────────
# GlobalAvgPool → Dense(256, ReLU) → Dropout(0.5) → Dense(8, Softmax)


def build_vgg16(num_classes=NUM_CLASSES, freeze_base=False):
    """
    Proposed VGG16 architecture (Fig.6):
      VGG16 base (ImageNet weights, classifier removed)
      → GlobalAveragePooling2D
      → Dense(256, activation=ReLU)
      → Dropout(0.5)
      → Dense(num_classes, activation=Softmax)
    """
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    if freeze_base:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace classifier with paper's head (uses GAP instead of flatten)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # GlobalAveragePooling2D
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512, 256),  # 512 = VGG16 last conv channels
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
    )
    return model


# ── EfficientNetB0 (Section 4.2, Fig.7) ──────────────────────────────────────
# GlobalAvgPool → Dropout(0.1) → Dropout(0.3) → Dense(8, Softmax)
# ALL layers unfrozen for end-to-end fine-tuning


def build_efficientnet_b0(num_classes=NUM_CLASSES, freeze_base=False):
    """
    Proposed EfficientNetB0 architecture (Fig.7):
      EfficientNetB0 base (ImageNet weights, ALL layers unfrozen)
      → GlobalAveragePooling2D
      → Dropout(0.1)   [first dropout in Fig.7]
      → Dropout(0.3)   [second dropout in Fig.7]
      → Dense(num_classes, activation=Softmax)

    Paper: "All layers were thawed to enable end-to-end fine-tuning"
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Paper: all layers unfrozen (freeze_base=False is the correct setting)
    if freeze_base:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace classifier head
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(0.1),  # Dropout(0.1) from Fig.7
        nn.Dropout(0.3),  # Dropout(0.3) from Fig.7
        nn.Linear(in_features, num_classes),
    )
    return model


# ── InceptionV3 (Section 4.3, Fig.8) ─────────────────────────────────────────
# GlobalAvgPool → Dropout(0.1) → Dense(128, ReLU) → Dropout(0.3) → Dense(8)


def build_inception_v3(num_classes=NUM_CLASSES, freeze_base=False):
    """
    Proposed InceptionV3 architecture (Fig.8):
      InceptionV3 base (ImageNet weights, aux_logits disabled)
      → GlobalAveragePooling2D
      → Dropout(0.1)
      → Dense(128, activation=ReLU)
      → Dropout(0.3)
      → Dense(num_classes, activation=Softmax)

    Note: InceptionV3 requires input size 299x299.
          The paper uses 224x224 for all models, so we set
          aux_logits=False and allow torchvision to handle resize.
    """
    model = models.inception_v3(
        weights=models.Inception_V3_Weights.IMAGENET1K_V1,
        aux_logits=False,
    )

    if freeze_base:
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    in_features = model.fc.in_features  # 2048
    model.fc = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(in_features, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes),
    )
    return model


# ── Model factory ─────────────────────────────────────────────────────────────

MODEL_BUILDERS = {
    "efficientnet_b0": build_efficientnet_b0,
    "resnet50": build_resnet50,
    "inception_v3": build_inception_v3,
    "vgg16": build_vgg16,
}

# Paper's expected accuracy results (Table 4) for reference
PAPER_RESULTS = {
    "efficientnet_b0": {"accuracy": 96.76, "precision": 96.84, "recall": 96.76, "f1": 96.77},
    "resnet50": {"accuracy": 93.51, "precision": 93.66, "recall": 93.51, "f1": 93.33},
    "inception_v3": {"accuracy": 93.51, "precision": 94.09, "recall": 93.51, "f1": 93.53},
    "vgg16": {"accuracy": 84.32, "precision": 84.57, "recall": 84.32, "f1": 83.15},
}


def build_model(model_name, num_classes=NUM_CLASSES, freeze_base=False, device=None):
    """
    Build and return model + device.

    Args:
        model_name  : one of 'efficientnet_b0', 'resnet50', 'inception_v3', 'vgg16'
        num_classes : number of output classes (default 8)
        freeze_base : freeze backbone during initial training phase
        device      : torch device (auto-detected if None)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name not in MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model '{model_name}'. " f"Choose from: {list(MODEL_BUILDERS.keys())}"
        )

    model = MODEL_BUILDERS[model_name](
        num_classes=num_classes,
        freeze_base=freeze_base,
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ref = PAPER_RESULTS.get(model_name, {})

    print(f"[Model] {model_name} | classes={num_classes} | device={device}")
    print(f"[Model] Params total={total:,}  trainable={trainable:,}")
    if ref:
        print(
            f"[Model] Paper target → acc={ref['accuracy']}%  "
            f"prec={ref['precision']}%  f1={ref['f1']}%"
        )

    return model, device


if __name__ == "__main__":
    for name in MODEL_BUILDERS:
        m, dev = build_model(name)
        # InceptionV3 needs 299x299; others 224x224
        sz = 299 if name == "inception_v3" else 224
        dummy = torch.randn(2, 3, sz, sz).to(dev)
        with torch.no_grad():
            out = m(dummy)
        print(f"  output shape: {out.shape}\n")
