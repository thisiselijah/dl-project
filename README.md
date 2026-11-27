# Deep Learning @NCKU

MTDDH 小兒髖關節關鍵點偵測作業：輸入單張正位骨盆 X 光，用 YOLO-Pose 預測左右髖關節共 8 個關鍵點，並以 Pose mAP50–95、NME、PCK@0.05 評估。

## 物件偵測基礎
物件偵測（Object Detection）結合了影像分類（Classification）與物件定位（Localization），目標是在一張輸入影像中找出所有感興趣的目標，預測其類別並以邊界框標示位置。

> **Problem Formulation.** Given an input image tensor $I \in \mathbb{R}^{H \times W \times C}$ of height $H$, width $W$, and $C$ channels, object detection is formulated as a set-valued mapping $f_\theta: \mathbb{R}^{H \times W \times C} \to \mathcal{P}_{\text{fin}}(\mathcal{B} \times \mathcal{C} \times \mathcal{S})$ that predicts a finite set of $N$ object instances:
> 
> $$\hat{\mathcal{D}} = \left\{ (\mathbf{b}_i, c_i, s_i) \right\}_{i=1}^{N}, \quad \mathbf{b}_i \in \mathcal{B},\; c_i \in \mathcal{C},\; s_i \in \mathcal{S}$$
> 
> 
> 
> where $\mathbf{b}_i = (x_i, y_i, w_i, h_i) \in \mathcal{B} \subset \mathbb{R}^4$ denotes the bounding box parameterized by its center coordinates $(x_i, y_i)$ and spatial dimensions $(w_i, h_i)$ (equivalently expressed via top-left and bottom-right corners $(x_{\min}, y_{\min}, x_{\max}, y_{\max})$), $c_i \in \mathcal{C} = \{1, 2, \dots, K\}$ represents the discrete semantic class index among $K$ foreground categories, and $s_i \in \mathcal{S} = [0, 1]$ indicates the predicted confidence score.

## Dataset
載點: [資料集](https://drive.google.com/file/u/0/d/1RiZPit24Z-WglDqzsDT5VUzkqWlScD-Q/view?usp=sharing&pli=1)

| cls | 類別 | kpt 0 | kpt 1 |
|---|---|---|---|
| 0 | hip_pelvis | 髖臼上外側緣 (ASM, P-line) | 三叉軟骨中心 (TCC, H-line) |
| 1 | hip_femour | 股骨頭中心 (FHC) | 幹骺端上緣中點 (MOFM, H-point) |

- 標註格式：`cls cx cy w h kx0 ky0 v0 kx1 ky1 v1`（座標正規化 0–1，v 為可見性）
- 每張影像通常 4 行：左右各一個 `hip_pelvis`、一個 `hip_femour`
- 切分：train 1041 / val 344 / test 347







<center>--- 以下是 AI 產生 ---</center>

## 開始步驟

### 1. 建立環境
```bash
# pyproject.toml 設定 requires-python = ">=3.12,<3.13"
uv python pin 3.12
rm -rf .venv
```

`xtcocotools`（mmpose 依賴）需要編譯，WSL / Ubuntu 先安裝 C 編譯器：
```bash
sudo apt update && sudo apt install -y build-essential
```

講義用 `pip install --no-deps` 裝 mmpose，是為了避開 `chumpy` 等無法 build 的依賴。uv 的對應設定已寫在 `pyproject.toml`：
```toml
[tool.uv]
exclude-dependencies = ["chumpy"]   # mmpose 依賴，評估用不到且無法 build

[tool.uv.extra-build-dependencies]
xtcocotools = ["numpy", "cython"]   # 套件 build 時需要但未宣告
```

clone 下來後直接同步即可（依賴已記錄在 `pyproject.toml` 與 `uv.lock`）：
```bash
uv sync
uv run python -c "from mmpose.evaluation.functional import keypoint_nme; print('ok')"
```

從零開始則用 `uv add`：
```bash
uv add ultralytics "mmengine>=0.4.0,<1.0.0" "mmcv-lite>=2.0.0,<3.0.0" "mmpose==1.3.2"
```

### 2. 準備資料
從上方連結下載，解壓到 `data/`：
```
data/
├─ data.yaml
├─ train/{images,labels}
├─ val/{images,labels}
└─ test/{images,labels}
```

> ⚠️ `data.yaml` 的 `path: .` 是相對於**執行指令的目錄**。若在 repo 根目錄執行，改成 `path: data`（或填絕對路徑）。

### 3. 訓練 baseline
```bash
uv run yolo pose train data=data/data.yaml model=yolo11n-pose.pt \
    imgsz=640 epochs=100 batch=16 project=runs name=baseline
```
- 預設 `fliplr=0.5` 可用：每個物件只含單側的點，`flip_idx: [0, 1]` 不需交換
- 可調整的方向：更大模型（s/m）、更大 `imgsz`（關鍵點精度對解析度敏感）、`mosaic`、`degrees`、`hsv_*`（X 光為灰階，色彩增強意義不大）
- 最佳權重：`runs/baseline/weights/best.pt`，記得複製到 `weights/`

### 4. 評估
- **Pose mAP50–95**：
  ```bash
  uv run yolo pose val data=data/data.yaml model=weights/best.pt split=test
  ```
- **NME / PCK@0.05**：寫腳本用 `mmpose.evaluation.functional` 的 `keypoint_pck_accuracy`、`keypoint_nme`
  1. 對 test 影像推論，取得預測框與關鍵點
  2. 將預測與 GT 物件配對（同類別 + 框中心最近，左右各一）
  3. `pred`, `gt`: `(N, 2, 2)` 像素座標；`mask`: `v > 0`；`norm`: GT 框長邊（像素）
  4. 未偵測到的 GT 物件要計入（例如給遠距離預測），不要直接跳過

### 5. 錯誤分析
- 依 NME 排序，挑誤差最大的幾張，把 GT（綠）與預測（紅）畫在影像上
- 分 Dataset 1（RGB JPG）/ Dataset 2（灰階 PNG）、分關鍵點類型統計誤差

## 繳交
- `<學號>_<姓名>.zip` 上傳 Moodle，內含：
  - 報告（PDF / Word）：模型選擇與理由、超參數（batch、lr、optimizer、loss、epochs）、前處理與資料增強、三項指標結果、錯誤分析
  - 程式碼（`.py` / `.ipynb`）
- 期末 Demo：助教會公布 GUI 程式與接口，需確認能讀取 `weights/` 中的權重並推論

## Checklist
- [ ] 環境建置（Python 3.12 + ultralytics + mmpose）
- [ ] 資料清理、`data.yaml` 路徑確認
- [ ] Baseline 訓練
- [ ] 評估腳本（mAP50–95 / NME / PCK@0.05）
- [ ] 調參 / 改進實驗
- [ ] 錯誤分析圖
- [ ] 報告
- [ ] Demo GUI 測試
