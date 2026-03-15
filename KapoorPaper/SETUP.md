# KapoorPaper API サーバー セットアップ手順

## 概要

EfficientNetB0 (Kapoor et al. 2025) による皮膚疾患8クラス分類モデルを
FastAPI サーバーとして起動し、帯状疱疹チェッカーアプリ (App.js) と連携します。

---

## 必要条件

- Python 3.10 以上
- 訓練済みチェックポイント: `checkpoints/best_efficientnet_b0.pth`

---

## 1. 依存パッケージのインストール

```bash
cd KapoorPaper
pip install -r requirements.txt
```

---

## 2. モデルの確認

`checkpoints/` に `best_efficientnet_b0.pth` が存在することを確認してください。

```bash
ls checkpoints/
# → best_efficientnet_b0.pth が表示されること
```

ファイルがない場合はトレーニングを実行:

```bash
python train.py \
  --model efficientnet_b0 \
  --train_dir skin-disease-dataset/train_set \
  --test_dir  skin-disease-dataset/test_set
```

---

## 3. API サーバーを起動

```bash
cd KapoorPaper
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

起動後、以下で動作確認:

```bash
curl http://localhost:8000/health
# → {"status":"ok","model":"efficientnet_b0", ...}
```

---

## 4. React Native アプリとの接続設定

`App.js` の先頭付近にある `AI_API_URL` を環境に合わせて変更してください:

| 環境 | 設定値 |
|---|---|
| iOS シミュレータ / Web | `http://localhost:8000` |
| Android エミュレータ | `http://10.0.2.2:8000` |
| 実機（同一Wi-Fi） | `http://<PCのIPアドレス>:8000` |

```js
// App.js 上部
const AI_API_URL = 'http://localhost:8000';  // ← ここを変更
```

---

## API エンドポイント

### `GET /health`
サーバーとモデルの稼働確認

### `POST /predict`
画像をアップロードして皮膚疾患を分類する

**リクエスト:** `multipart/form-data` — `file` フィールドに画像 (JPEG/PNG)

**レスポンス例:**
```json
{
  "predicted_class": "VI-shingles",
  "predicted_label_ja": "帯状疱疹（ウイルス性）",
  "confidence": 94.3,
  "shingles_detected": true,
  "all_probs": {
    "VI-shingles": 94.3,
    "VI-chickenpox": 3.1,
    "BA-cellulitis": 1.2,
    ...
  }
}
```

---

## AI スコア加算ルール（App.js）

| AI検出結果 | 確信度 | スコア加算 |
|---|---|---|
| 帯状疱疹 (VI-shingles) | 70% 以上 | **+3** |
| 帯状疱疹 (VI-shingles) | 50〜70% | **+2** |
| 帯状疱疹 (VI-shingles) | 50% 未満 | **+1** |
| 水痘 (VI-chickenpox) | — | **+1** |
| その他 | — | 0 |

---

## 分類クラス（8クラス）

| クラス名 | 日本語 |
|---|---|
| BA-cellulitis | 蜂窩織炎（細菌性） |
| BA-impetigo | とびひ（細菌性） |
| FU-athlete-foot | 水虫（真菌性） |
| FU-nail-fungus | 爪白癬（真菌性） |
| FU-ringworm | 白癬・たむし（真菌性） |
| PA-cutaneous-larva-migrans | 皮膚幼虫移行症（寄生虫） |
| VI-chickenpox | 水痘・水ぼうそう（ウイルス性） |
| VI-shingles | **帯状疱疹（ウイルス性）** ← 主対象 |
