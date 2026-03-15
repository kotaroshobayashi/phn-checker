"""
FastAPI Inference Server – Skin Disease Multi-Classification
Kapoor et al. (2025) / EfficientNetB0

起動方法:
    cd KapoorPaper
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

エンドポイント:
    GET  /health    → 稼働確認
    POST /predict   → 画像アップロード → 皮膚疾患分類結果を返す
"""

from contextlib import asynccontextmanager
import io
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from models import CLASS_NAMES, build_model
from dataset import get_val_transforms

# ── パス設定 ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
CHECKPOINT  = BASE_DIR / "checkpoints" / "best_efficientnet_b0.pth"
MODEL_NAME  = "efficientnet_b0"
INPUT_SIZE  = 224

# ── グローバルモデル（起動時にロード） ────────────────────────────────────────
_model  = None
_device = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """サーバー起動時にモデルをロード、シャットダウン時に解放"""
    global _model, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not CHECKPOINT.exists():
        raise RuntimeError(
            f"チェックポイントが見つかりません: {CHECKPOINT}\n"
            "先に train.py で学習してください:\n"
            "  python train.py --model efficientnet_b0 "
            "--train_dir skin-disease-dataset/train_set "
            "--test_dir skin-disease-dataset/test_set"
        )

    ckpt = torch.load(CHECKPOINT, map_location=_device, weights_only=True)
    _model, _device = build_model(MODEL_NAME, device=_device)
    _model.load_state_dict(ckpt["model"])
    _model.eval()

    best_acc = ckpt.get("best_acc", "?")
    print(f"[API] モデルロード完了: {MODEL_NAME} | val_acc={best_acc}% | device={_device}")

    yield  # サーバー稼働中

    _model = None
    print("[API] モデルを解放しました")


# ── FastAPI アプリ ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="皮膚疾患分類 API",
    description=(
        "EfficientNetB0 による皮膚疾患8クラス分類 (Kapoor et al. 2025)\n\n"
        "分類クラス: BA-cellulitis / BA-impetigo / FU-athlete-foot / "
        "FU-nail-fungus / FU-ringworm / PA-cutaneous-larva-migrans / "
        "VI-chickenpox / VI-shingles"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 開発用。本番では origin を制限すること
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── エンドポイント ─────────────────────────────────────────────────────────────

@app.get("/health", summary="稼働確認")
def health():
    """サーバーとモデルの稼働状態を返す"""
    return {
        "status":  "ok",
        "model":   MODEL_NAME,
        "device":  str(_device),
        "classes": CLASS_NAMES,
    }


@app.post("/predict", summary="皮膚疾患分類")
async def predict(file: UploadFile = File(..., description="分類する皮膚画像（JPEG/PNG）")):
    """
    画像ファイルを受け取り、EfficientNetB0 で皮膚疾患を分類する。

    **Returns**
    - `predicted_class`   : 最も確率の高い疾患クラス名（英語）
    - `predicted_label_ja`: 日本語ラベル
    - `confidence`        : 予測確率（0〜100%）
    - `shingles_detected` : 帯状疱疹（VI-shingles）が最上位クラスなら True
    - `all_probs`         : 全クラスの確率 dict（降順ソート済み）
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="モデルが未ロードです")

    # ── 画像読み込み ──────────────────────────────────────────────────────────
    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="空のファイルです")

    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"画像の読み込みに失敗しました: {exc}")

    # ── 前処理 → 推論 ─────────────────────────────────────────────────────────
    transform = get_val_transforms(INPUT_SIZE)
    tensor = transform(img).unsqueeze(0).to(_device)

    with torch.no_grad():
        logits = _model(tensor)
        if isinstance(logits, tuple):
            logits = logits[0]

    probs    = F.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = int(probs.argmax())
    pred_cls = CLASS_NAMES[pred_idx]
    conf     = float(probs[pred_idx]) * 100

    # 全クラス確率（降順）
    all_probs = {
        CLASS_NAMES[i]: round(float(probs[i]) * 100, 2)
        for i in range(len(CLASS_NAMES))
    }
    all_probs_sorted = dict(
        sorted(all_probs.items(), key=lambda x: -x[1])
    )

    return {
        "predicted_class":    pred_cls,
        "predicted_label_ja": _CLASS_LABELS_JA.get(pred_cls, pred_cls),
        "confidence":         round(conf, 2),
        "shingles_detected":  pred_cls == "VI-shingles",
        "all_probs":          all_probs_sorted,
    }


# ── 日本語ラベル（レスポンスにも含める） ──────────────────────────────────────
_CLASS_LABELS_JA: dict[str, str] = {
    "BA-cellulitis":              "蜂窩織炎（細菌性）",
    "BA-impetigo":                "とびひ（細菌性）",
    "FU-athlete-foot":            "水虫（真菌性）",
    "FU-nail-fungus":             "爪白癬（真菌性）",
    "FU-ringworm":                "白癬・たむし（真菌性）",
    "PA-cutaneous-larva-migrans": "皮膚幼虫移行症（寄生虫）",
    "VI-chickenpox":              "水痘・水ぼうそう（ウイルス性）",
    "VI-shingles":                "帯状疱疹（ウイルス性）",
}
