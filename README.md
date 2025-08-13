# Impact of variation in tissue staining and scanning devices on performance of pan-cancer AI models: a study of sarcoma and their mimics

## Introduction
Microscopic analysis of histopathology is considered the gold standard for cancer diagnosis and prognosis. Recent advances in AI, driven by large-scale digitisation and pan-cancer foundation models, are opening new opportunities for clinical integration. However, it remains unclear how robust these foundation models are to real-world sources of variability, particularly in staining and scanning protocols.

In this study, we use soft tissue tumours, a rare and morphologically diverse tumour type, as a challenging test case to systematically investigate the colour-related robustness and generalisability of seven AI models. Controlled staining and scanning experiments are designed to assess model performances across diverse real-world data sources. Foundation models, particularly **UNI-v2**, **Virchow**, and **TITAN**, demonstrated encouraging robustness to staining and scanning variation, particularly when a small number of stain-varied slides were included in the training loop, highlighting their potential as adaptable and data-efficient tools for real-world digital pathology workflows.

---

## Repository Structure

```
analysis_scripts/         # Statistical analysis scripts
utils/                    # Helper functions and classes
visualisation_scripts/    # Plot and interactive plot generation scripts
.gitignore
data_split.py             # Train/val/test splitting with class balancing
extract_embedding.py      # Embedding extraction (Google Path Foundation only)
train.py                  # Train an attention-based MIL model
test.py                   # Test a trained MIL model
trident_with_normalisation.py  # Optional setting: apply colour normalisation before embedding extraction using **Trident**

```

---

## Modules & Usage

### 1) `data_split.py`
Perform a **train/validation/test split** on a CSV with two columns: `case_id` and `ground_truth`. The split is **stratified per class** to preserve distribution.

**Parameters**
- `source_csv` *(str)* — Path to input CSV (`case_id`, `ground_truth`).
- `train_size` *(int)* — Training percentage (0–100).
- `val_size` *(int)* — Validation percentage (0–100).
- `test_size` *(int)* — Testing percentage (0–100).
- `output_folder` *(str)* — Folder to save split CSVs.

**Example Usage**
```bash
python data_split.py --source_csv /path/to/cases.csv --train_size 60 --val_size 20 --test_size 20 --output_folder /path/to/output_splits
```

---

### 2) Embedding extraction
We use **Trident** for embedding extraction for **UNI-v2**, **CONCH-v1.5**, **TITAN**, **Virchow**, **PRISM**, and **ResNet-CNN**, for more information about **Trident**, please refer to the **Useful Links** section below.

We have a bespoke script `extract_embedding.py` for tile-level embedding extraction of the **Google Path Foundation** model (TensorFlow + GPU).

**Parameters**
- `wsi_path` *(str)* — Folder containing WSIs.
- `tile_path` *(str)* — Folder containing `.h5` tiling info (CLAM format).
- `model_path` *(str)* — Path to the Path Foundation model.
- `output_path` *(str)* — Output folder for generated embeddings.

**Example Usage**
```bash
python extract_embedding.py --wsi_path /path/to/wsi_folder --tile_path /path/to/h5_tiles --model_path /path/to/path_foundation_model --output_path /path/to/output_embeddings
```

> We offer an optional setting by applying **colour normalisation** before tile-level embedding extraction with **Trident** using `trident_with_normalisation.py`. This script works as a bridge without modifying **Trident** itself, **cuCIM** and **torch-staintools** are used so this script is fully GPU-accelerated without the I/O bottleneck.
> **Parameters**
> - `wsi_dir` *(str)* — Directory containing WSIs.
> - `coords_dir` *(str)* — Directory of `*_patches.h5` coordinate files.
> - `out_dir` *(str)* — Directory to save output `.h5` feature files.
> - `target_img_path` *(str)* — Path to the target image for normaliser fitting.
> - `encoder_name` *(str)* — Trident patch encoder (e.g., `uni_v1`, `uni_v2`, `conch_v1`).
> - `norm_method` *(str)* — One of `vahadane`, `macenko`, `reinhard`.
> - `batch_size` *(int)* — Patch batch size.
> - `gpu` *(int)* — CUDA device index.
> - `mag` *(int)* — Desired magnification (e.g., `20` for 20×).
> - `custom_list_of_wsis` *(str)* — Optional CSV with header `wsi` and rows like `XXX.svs` to **filter** which slides to process.

**Example Usage**
```bash
python trident_with_normalisation.py --wsi_dir /path/to/wsis --coords_dir /path/to/coords --out_dir /path/to/output --target_img_path /path/to/target_image.tif --encoder_name uni_v2 --norm_method macenko --batch_size 128 --gpu 0 --mag 20 --custom_list_of_wsis /path/to/wsi_list.csv
```

---

### 3) `train.py`
Train an **attention-based Multi-Instance Learning (MIL)** classifier. Slides (bags) contain variable numbers of patch embeddings (instances); attention aggregates to a slide-level representation.

**Features**
- Multiple training folders supported.
- Validation after each epoch; **best model is saved** by validation accuracy.
- **Early stopping** based on validation loss with `patience`.
- Works with both **CSV** and **h5** embeddings (`emb_type`).

**Parameters**
- `train_folder`, `train_folder_2`, `train_folder_3` *(str)* — Training data folders (optional 2 & 3).
- `train_labels` *(str)* — Training label CSV.
- `val_folder` *(str)* — Validation data folder.
- `val_labels` *(str)* — Validation label CSV.
- `model_folder` *(str)* — Output folder for checkpoints and logs.
- `k_instances` *(int)* — Number of instances sampled per bag.
- `epochs` *(int)* — Number of epochs.
- `lr` *(float)* — Learning rate.
- `patience` *(int)* — Early stopping patience.
- `num_class` *(int)* — Number of classes.
- `emb_type` *(str)* — Choose `h5` or `csv`.

**Example Usage**
```bash
python train.py --train_folder /path/to/train_data --train_labels /path/to/train_labels.csv --val_folder /path/to/val_data --val_labels /path/to/val_labels.csv --model_folder /path/to/save_model --k_instances 500 --epochs 200 --patience 10 --num_class 14 --emb_type h5
```

---

### 4) `test.py`
Evaluate a trained **MIL classifier** on a held-out test set.

**Parameters**
- `test_folder` *(str)* — Test data folder.
- `test_labels` *(str)* — Test label CSV (with `case_id`, `ground_truth`).
- `model` *(str)* — Path to saved model checkpoint.
- `output` *(str)* — Output folder for metrics and predictions.
- `emb_type` *(str)* — Choose `h5` or `csv`.

**Example Usage**
```bash
python test.py --test_folder /path/to/test_data --test_labels /path/to/test_labels.csv --model /path/to/saved_model.pth --output /path/to/output_predictions --emb_type h5
```

---


## Useful Links
- **Trident*: https://github.com/mahmoodlab/TRIDENT/
- **cuCIM**: https://github.com/rapidsai/cucim
- **torch-staintools**: https://github.com/CielAl/torch-staintools

---

## Citation
If you use this repository or code in your research, please cite (TBC, under preparation)

