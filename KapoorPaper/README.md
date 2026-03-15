# 皮膚疾患多クラス分類 CNN

**論文再現実装**: Kapoor et al. (2025)  
*"Convolutional Neural Network-based Multi-Classification of Skin Disease with Fine-Tuned ResNet50 and VGG16"*  
The Open Bioinformatics Journal, Vol. 18, 2025

---

## 論文の手法と実装の対応

| 論文の記述 | 実装箇所 | 詳細 |
|---|---|---|
| 4モデル（ResNet50, VGG16, EfficientNetB0, InceptionV3）| `models.py` | 各アーキテクチャ（Fig.5-8）を再現 |
| ResNet50 head: GAP→Dense(512,ReLU)→Dense(8) | `build_resnet50()` | Fig.5 |
| VGG16 head: GAP→Dense(256,ReLU)→Dropout(0.5)→Dense(8) | `build_vgg16()` | Fig.6 |
| EfficientNetB0 head: GAP→Dropout(0.1)→Dropout(0.3)→Dense(8) + 全層解凍 | `build_efficientnet_b0()` | Fig.7 |
| InceptionV3 head: GAP→Dropout(0.1)→Dense(128,ReLU)→Dropout(0.3)→Dense(8) | `build_inception_v3()` | Fig.8 |
| Adam optimizer, batch=32, epochs=20 | `train.py` | Section 3 |
| Accuracy/Precision/Recall/F1 (Eq.1-4) | `compute_metrics()` | Section 5 |
| クラス重み付け（不均衡対策） | `compute_class_weights()` | Section 3.1 |
| 224×224px リサイズ（InceptionV3は299×299） | `dataset.py` | Section 3.1 |

### 論文の報告値（Table 4）

| モデル | 精度 | 適合率 | 再現率 | F1スコア |
|---|---|---|---|---|
| **EfficientNetB0** | **96.76%** | **96.84%** | **96.76%** | **96.77%** |
| ResNet50 | 93.51% | 93.66% | 93.51% | 93.33% |
| InceptionV3 | 93.51% | 94.09% | 93.51% | 93.53% |
| VGG16 | 84.32% | 84.57% | 84.32% | 83.15% |

---

## ファイル構成

```
skin_disease/
├── models.py         # 4つのCNNアーキテクチャ定義（Fig.5-8）
├── dataset.py        # KaggleデータセットのDataLoader（前処理・クラス重み）
├── train.py          # 学習スクリプト（Adam・メトリクス計算）
├── infer.py          # 推論・4モデル比較（Table 4再現）
├── requirements.txt  # 依存ライブラリ
└── README.md         # 本ファイル
```

---

## セットアップ

```bash
# 1. インストール
pip install -r requirements.txt

# 2. GPU確認
python -c "import torch; print(torch.cuda.is_available())"
```

---

## データセット取得

論文で使用したKaggleデータセットをダウンロードします：
https://www.kaggle.com/datasets/subirbiswas19/skin-disease-dataset

ダウンロード後の構造：
```
skin-disease-dataset/
├── train/
│   ├── BA- cellulitis/      (136枚)
│   ├── BA-impetigo/         (80枚)
│   ├── FU-athlete-foot/     (124枚)
│   ├── FU-nail-fungus/      (129枚)
│   ├── FU-ringworm/         (90枚)
│   ├── PA-cutaneous-larva-migrans/ (100枚)
│   ├── VI-chickenpox/       (136枚)
│   └── VI-shingles/         (130枚)
└── test/
    └── (同じ8フォルダ、各クラス20-34枚)
```

---

## 学習

### 単一モデル（例: EfficientNetB0）

```bash
python train.py \
    --model efficientnet_b0 \
    --train_dir skin-disease-dataset/train \
    --test_dir  skin-disease-dataset/test \
    --output_dir checkpoints \
    --epochs 20 \
    --batch_size 32 \
    --lr 0.001
```

### 4モデルを順番に全て学習

```bash
for MODEL in efficientnet_b0 resnet50 inception_v3 vgg16; do
    python train.py \
        --model $MODEL \
        --train_dir skin-disease-dataset/train \
        --test_dir  skin-disease-dataset/test \
        --output_dir checkpoints \
        --epochs 20 \
        --batch_size 32
done
```

学習中の表示例：
```
Epoch 18/20 | 45.2s
  TRAIN  loss=0.1823  acc=97.51%  prec=97.62%  rec=97.51%  f1=97.55%
  TEST   loss=0.1241  acc=96.58%  prec=96.71%  rec=96.58%  f1=96.63%
  ✓ Best checkpoint saved → acc=96.58%
```

---

## 推論

### 単一画像の分類

```bash
python infer.py predict \
    --checkpoint checkpoints/best_efficientnet_b0.pth \
    --image my_skin_image.jpg
```

出力例：
```
── Prediction ─────────────────────────────
  Image     : my_skin_image.jpg
  Prediction: VI-chickenpox
  Confidence: 94.32%

── All class probabilities ────────────────
  VI-chickenpox                  94.32%  ██████████████████
  VI-shingles                     3.12%  
  BA-cellulitis                   1.05%  
  ...
```

### 4モデル比較表（Table 4の再現）

```bash
# 全4モデルのチェックポイントが必要
python infer.py compare \
    --checkpoint_dir checkpoints \
    --test_dir skin-disease-dataset/test
```

出力例（論文Table 4に相当）：
```
===================================================================
 Parameter       VGG-16    InceptionV3  EfficientNetB0    ResNet50
───────────────────────────────────────────────────────────────────
 Accuracy        84.32%       93.51%         96.76%        93.51%
 Precision       84.57%       94.09%         96.84%        93.66%
 Recall          84.32%       93.51%         96.76%        93.51%
 F1              83.15%       93.53%         96.77%        93.33%
===================================================================
```

---

## 注意事項

- 本実装は**研究・学習目的**です。医療診断には使用しないでください。
- InceptionV3のみ入力サイズが299×299になります（他は224×224）。
- `--freeze_base` オプションで骨格ネットワークを凍結できます（論文はデフォルトで非凍結）。
- EfficientNetB0は論文通り全層を解凍してエンドツーエンドでファインチューニングします。
