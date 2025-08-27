## Multi-Sensor Deforestation Detection (S1 + S2)

**Goal:** Detect deforestation by **fusing optical (Sentinel-2)** and **SAR (Sentinel-1)**, handling real-world issues (clouds, missing data, mixed providers) and evaluating against single-sensor baselines.

This repo contains:

* A **multi-sensor dataset loader** (cloud masking, temporal pairing, harmonization, quality scoring).
* A **PyTorch Lightning** pipeline with **segmentation\_models\_pytorch (SMP)** UNet for semantic segmentation.
* **EDA utilities** and **notebooks** for sanity checks and qualitative analysis.
* **W\&B integration** (optional) to track metrics and validation panels. [**$\rightarrow$Link$\leftarrow$**](https://wandb.ai/scigeo/capacity-defor?nw=nwuserryali)

---

## TL;DR (Quick Start)

```bash
# 1) Create env (conda recommended)
conda create -n ms-defor python=3.10 -y
conda activate ms-defor
pip install -r requirements.txt  # or pip install -e . if you package it
cd src

# 2) Check config.yaml (sensors, temporal policy, training, wandb)

# 3) Train
python -m pl_train

# 4) Notebooks
# (open notebooks/pipeline.ipynb or notebooks/inference.ipynb)

# 5) Inference in a notebook (load ckpt) These will be uploaded to HuggingFace 
# see notebooks/inference.ipynb sample cells
```
<!-- > [!WARNING]
> The `requirements.txt` is not completed -->

---

## Repository Structure

```
.
|-- config.yaml
|-- dataset/                         # alerts (one folder per alert_id)
|-- docs/                            # dataset docs (provided)
|-- figs/                            # quick panels exported by notebooks
|-- notebooks/
|   |-- pipeline.ipynb               # end-to-end demo
|   |-- inference.ipynb              # experiment inference
|   |-- starter-dataloader.ipynb     # basic data exploration (provided)
|-- src/
|   |-- config.py                    # dataclasses & YAML binding
|   |-- dataset.py                   # MultiSensorAlertDataset (S1/S2) [support other databases]
|   |-- sentinel1.py                 # Sentinel-1 adapter (VV/VH, indices)
|   |-- sentinel2.py                 # Sentinel-2 adapter (bands, indices, SCL mask)
|   |-- temporal.py                  # pairing policies (closest_pair, expand_pairs)
|   |-- raster.py                    # reprojection/cropping helpers
|   |-- stats.py                     # per-provider band stats
|   |-- eda.py                       # EDA plots & dataset health checks
|   |-- metrics.py                   # IoU_deforest, F1/P/R, per-alert IoU
|   |-- pl_data.py                   # LightningDataModule (splits, ChannelSelector)
|   |-- pl_module.py                 # LitSeg (SMP Unet, losses, logging)
|   |-- pl_utils.py                  # W&B images, panels
|   |-- pl_train.py                  # Lightning Trainer entrypoint
|   |-- utils.py, __init__.py
|   `-- runs/                        # checkpoints (best.ckpt, last.pth)
`-- wandb/                           # W&B run artifacts (optional)
```

---

## Dataset Layout

Each `dataset/<alert_id>/` contains imagery for a 4×4 km tile around the alert and a ground-truth mask:

```
dataset/
  ├── 1387923/
  │   ├── 1387923_20250301_mask.tif           # GT (0=forest, 255=defor)
  │   ├── sentinel2/*.tif + *_bands.json + *_metadata.json + *_ndvi.tif ...
  │   ├── sentinel1/*.tif + *_bands.json + *_metadata.json + *_rvi.tif ...
  │   ├── landsat/...
  │   └── cbers4a/...
  └── ...
```

Masks are reprojected to the **primary grid** (S2 if present, else S1) with `nearest` resampling.

---

## Configuration (config.yaml)

All components read from a single YAML. Key blocks:

```yaml
root_dir: ../dataset
seed: 42

sensors:
  sentinel2:
    enabled: true
    bands: [B02, B03, B04, B08, B11, B12]
    indices: []             # optional [NDVI, NDWI]
    provider_norm: true     # robust per-provider normalization
    scl_cloud_mask:
      use: true
      classes_mask_aws:     [11, 12, 30, 31, 34, 35, 38, 43]
      classes_mask_bdc_l2a: [11, 12, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 42, 43, 47]
      classes_mask_bdc_l6d: [3, 8, 9, 10, 11, 12, 33, 38]
      dilate_px: 1          # safe default for cloud fringes
  sentinel1:
    enabled: true
    bands: [vv, vh]
    indices: []              # [RVI, RFDI]
    add_ratio: false
    add_rfdi: true
    add_rvi:  true

temporal:
  mode: single
  pre_days: 30
  post_days: 14
  selector: closest_post
  pairing: expand_pairs       # closest_pair | expand_pairs
  max_pair_delta_days: 8
  expand_pairs: true

sampling:
  crop_size: 448
  center_crops: true

stats:
  p_low: 1
  p_high: 99
  cache_path: stats/ms_stats.json

training:
  experiment: s2idxs1idx      # s2 | s2s1 | s2s1idx | s1 | s1idx | s2idx ...
  min_clean: 0.6              # per-sample clean_percent filter
  batch_size: 6
  epochs: 40
  val_frac: 0.3
  encoder: resnet34
  encoder_weights: imagenet
  lr: 1.0e-3
  threshold: 0.2
  early_stopping_patience: 10
  reduce_on_plateau:
    factor: 0.5
    patience: 3
  mask_positive_value: 255    # GT uses 255 as “deforestation”
  mask_ignore_values: []      # add nodata here if present

wandb:
  enabled: true
  project: capacity-defor
  name: s2idxs1idx_wce_v2
  mode: online
  tags: ["baseline","s2idxs1idx","bs6","vf0.3","th0.2","40e"]
```

> [!NOTE] 
> **Sensor gating:** setting `sensors.sentinel1.enabled: false` (or Sentinel-2) excludes that sensor entirely **at indexing time** (dataset won’t create records that lack the enabled sensors when `experiment` demands them).

---

## Data Pipeline

### Adapters

* **Sentinel-2** (`sentinel2.py`): reads bands, optional indices (NDVI/NDWI), applies **SCL cloud mask** with provider-specific classes, and robust per-provider normalization based on cached percentiles (1–99).
* **Sentinel-1** (`sentinel1.py`): reads VV/VH (optionally converts to dB), adds **RVI/RFDI/ratio** if requested.

### Temporal pairing (`temporal.py`)

* **closest\_pair**: pick one S2 and nearest S1 in time, within `max_pair_delta_days`.
* **expand\_pairs**: create **multiple pairs** per alert across the window (good for change analysis). Falls back to per-sensor closest if pairing fails.

> [!WARNING]
> This section was not fully tested


### Harmonization & Validity (`raster.py`, `dataset.py`)

* Reprojects all channels to the **primary grid** (S2 preferred).
* Builds **validity masks**:

  * `valid_inter` (intersection of per-sensor valid pixels),
  * `valid_union` (union),
  * **dynamic brightness gate** for S2 RGB to avoid “all-black but clean” artifacts.
* Outputs per-record **quality** (clean\_percent, per-sensor clean, finite\_percent).

---

## EDA

Open `notebooks/pipeline.ipynb` to:

* Inspect **pairing histograms** (Δdays).
* Visualize **valid coverage** (`clean_percent`) distributions and thresholds.
* Render **quick panels** (S2 RGB/NDVI, S1 VV/VH/RVI, mask).

Functions provided in `src/eda.py`:

* `summarize_dataset(ds)`
* `plot_pairing_hist(ds)`: Dates difference between S1 and S2
* `plot_valid_hist(ds)`: Valid fraction by sample (QA/clouds)
* `band_stats_by_provider(ds)`: Mainly stats by provider
* `quick_panel(sample, output=None)`: Show a plot image for sample

> [!NOTE]
> The `figs` folder contains some examples with images created by `quick_panel` function. It was used to verify the image quality of all datasets (S2 and S1).

---

## Training (PyTorch Lightning + SMP UNet)

* **Model:** `src/pl_module.py::LitSeg` (UNet encoder configurable; 1-channel sigmoid head).
* **Loss:** **Weighted BCE + Dice** (auto `pos_weight` per batch from valid pixels) to counter class imbalance.
* **Metrics:** IoU (deforestation class), F1, Precision/Recall, **mean per-alert IoU**.
* **Logging:** `pl_utils.py` logs **validation panels to W\&B** (RGB, GT, Pred) each epoch.

Run:

```bash
python -m src.pl_train
```

Outputs:

* `src/runs/<run_name>/best.ckpt` (Lightning checkpoint)
* `last.pth` (state\_dict for convenience)

---

## Inference

Notebook-friendly helpers:

* Load db & model:

  ```python
  from infer_utils import load_db, load_model_from_ckpt
  cfg, dm   = load_db("../config.yaml")
  model     = load_model_from_ckpt("../config.yaml", "src/runs/.../best.ckpt")
  ```
* Select **same alert\_id across dates**:

  ```python
  from infer_utils import indices_by_alert_sorted, infer_on_subset_index
  val_subset = dm.val_dataloader().dataset
  idxs = indices_by_alert_sorted(dm, split="val", alert_id="1387925", sensor_preference="S2", max_items=4)
  for j in idxs:
      prob, pred, s = infer_on_subset_index(val_subset, model, j, thr=None, show=True)
  ```
* Internally we **reorder channels** to match model’s `hparams.channels`.

---

## Experiments & Naming

Checkpoints under `src/runs/` follow run names, e.g.:

* `ms-defor-v1_s2`             $\rightarrow$  S2 only
* `ms-defor-v1_s2s1idx`        $\rightarrow$  S2 + S1 indices
* `s1_wce_v1`                  $\rightarrow$  S1 only with weighted loss
* `s2idxs1idx_wce_v2`          $\rightarrow$  current fused setup with indices

> Use `training.experiment` in YAML to activate specific channel presets (`s2`, `s1`, `s2s1`, `s2idx`, `s2s1idx`, …) or define an explicit channel list if you prefer.

---

## Results (example placeholders)

* **Single-sensor baselines**:

  * S2 only (RGB+NIR or RGB+SWIR / +NDVI)
  * S1 only (VV,VH / +RVI)
* **Fusion**:

  * Early fusion (concat channels)
  * *(Optional)* dual-branch late fusion

**Report** key tables/figures:

* IoU\_deforest / F1 on **val** for each setting.
* **Per-alert IoU** distribution (boxplot/hist).
* Δdays pairing histogram, clean\_percent distribution.
* Qualitative panels for representative alerts (varied cloud cover).

> [!IMPORTANT]
> To see the experiments and results, click on the following [**link**](https://wandb.ai/scigeo/capacity-defor?nw=nwuserryali)

---

## Reproducibility

* `seed` in YAML for Python/NumPy/Torch seeds.
* **No leakage across splits**: splitting by **alert\_id** in `pl_data.py`.
* Log config and git hash in W\&B (optional).
  
---

<!-- # Mini-Report: Outline & Approach

A concise (2–4 pages) write-up that directly addresses the brief.

## 1. Problem & Data

* **Motivation:** optical S2 rich but cloudy; SAR S1 cloud-proof but different physics; goal is fusion.
* **Dataset:** 100 alerts (4×4 km), S1/S2 (+Landsat/CBERS), 30d pre / 14d post, GT masks (deforest=255).
* **Challenges:** clouds, missing acquisitions, provider heterogeneity, temporal uncertainty (±1–2 weeks).

## 2. Methodology

### 2.1 Preprocessing & Harmonization

* Per-provider normalization (robust 1–99 percentiles).
* S2 **SCL cloud masking** with provider-specific class sets (+ 1-px dilation).
* Dynamic **brightness gating** to avoid “black but clean” tiles.
* Reprojection to a single **primary grid** (S2 preferred), consistent crops (448×448).

### 2.2 Temporal Handling

* **Pairing policy:** `closest_pair` (one S2 + nearest S1) and `expand_pairs` (multiple pairs within ±window).
* Report **Δdays histogram** and `clean_percent` distribution.
* *(Optional)* Temporal deltas: add channels `[S2_post – S2_pre]`, `[S1_post – S1_pre]`, or a Siamese UNet for change.

### 2.3 Sensor Fusion

* **Early fusion** baseline: concatenate channels (e.g., `S2: B02,B03,B04,B08,B11,B12,+NDVI | S1: VV,VH,+RVI/RFDI`).
* *(Optional)* Late fusion: two encoders (S1/S2) → feature concat → decoder (report if implemented).
* Loss: **Weighted BCE + Dice** (auto `pos_weight` from valid pixels), handles strong class imbalance.

### 2.4 Training & Splits

* Split by **alert\_id** (no leakage).
* Hyperparams (config.yaml): encoder, lr, batch size, threshold for metrics.
* Early stopping on `val/iou_def` or `val/loss`, ReduceLROnPlateau.

## 3. Experiments

### 3.1 Baselines

* **S2 only** (w/ or w/o indices).
* **S1 only** (w/ or w/o indices).
* Training settings aligned across runs.

### 3.2 Fusion

* **S2 + S1 (early fusion)** (+ indices).
* *(Optional)* Fusion variants (e.g., late fusion two-branch).

### 3.3 Ablations / Sensitivity

* Effect of `min_clean` (data quality gate).
* Effect of `max_pair_delta_days`.
* Effect of class weighting (WBCE+Dice vs Dice-only).

## 4. Results

**Quantitative** (validation set):

* **IoU\_deforest**, **F1**, **Precision/Recall**, **mean per-alert IoU** (primary).
* Table comparing **S2 only**, **S1 only**, **S2+S1** (bold best).
* *(Optional)* Confidence intervals (bootstrapping over alerts).

**Qualitative**:

* 3–5 **panels** (RGB/GT/Pred) across different cloud conditions.
* For 1–2 alerts, show **multiple dates** to illustrate temporal sensitivity.

**Finding:** Fusion improves robustness under cloud and noise scenarios (explain where it helps most).

## 5. Discussion

* **Multi-Modal Fusion:** direct concatenation works surprisingly well; S1 helps when S2 is clouded or spectrally ambiguous.
* **Temporal Design:** with sparse/irregular timestamps, pairing to nearest images yields workable change signals; future: Siamese or contrastive training with explicit pre/post.
* **Data Messiness:** provider class maps, invalid/NoData, and brightness gate are critical to prevent mislabeled “clean” tiles.
* **Limitations:** small dataset (100 alerts), label timing uncertainty, limited CBERS/Landsat usage in v1.

## 6. What’s Next

* Two-branch encoders (S1/S2) with learned fusion.
* Explicit **change-aware** architectures (Siamese UNet, temporal attention).
* Hard-example mining, more aggressive cloud modeling (S2 cloud prob., shadow detection).
* Calibrate decision threshold; post-processing (morphology) for region continuity.

## 7. Reproducibility & Repo

* How to run (train/infer), checkpoints path, config diffs for each run.
* Link to W\&B tables and panels, commit hash of best runs. -->
