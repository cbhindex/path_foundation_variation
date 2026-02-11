# Impact of tissue staining and scanner variation on the performance of pathology foundation models: a study of sarcomas and their mimics

## Introduction
Microscopic analysis of histopathology is considered the gold standard for cancer diagnosis and prognosis. Recent advances in AI, driven by large-scale digitisation and pan-cancer foundation models, are opening new opportunities for clinical integration. However, it remains unclear how robust these foundation models are to real-world sources of variability, particularly in staining and scanning protocols.

In this study, we use soft tissue tumours, a rare and morphologically diverse tumour type, as a challenging test case to investigate the colour-related robustness and generalisability of seven AI models. Controlled staining and scanning experiments were used to assess model performance across diverse real-world data sources. Foundation models, particularly **UNI-v2**, **Virchow**, and **TITAN**, demonstrated encouraging robustness to staining and scanning variation, especially when a small number of stain-varied slides were included in the training loop.

Project links:
- Dataset portal: https://bhchai.com/visualise_scan_stain_efffects/dataset.html
- Project website: https://cbhindex.github.io/visualise_scan_stain_efffects
- Preprint: https://www.biorxiv.org/content/10.1101/2025.08.18.670932v2

---

## Repository Structure

```text
analysis_scripts/                Statistical analysis scripts (bootstrap, hospital-wise metrics, etc.)
utils/                           Helper functions and classes for PyTorch/TensorFlow workflows
visualisation_scripts/           Visualisation utilities (heatmaps, radar plots, t-SNE)

data_split.py                    Train/validation/test splitting with class balancing
extract_embedding.py             Google Path Foundation tile embedding extraction (TensorFlow)
trident_with_normalisation.py    Optional stain normalisation + Trident embedding extraction
train.py                         Attention-based MIL model training
test.py                          Attention-based MIL model evaluation
run_logistic_regression.py       Slide-level Logistic Regression evaluation (TITAN/PRISM embeddings)

environment.base.yml             Cross-platform base Conda environment definition
environment.mac.yml              macOS Conda overlay (PyTorch/torchvision)
environment.linux-cuda.yml       Linux CUDA Conda overlay (PyTorch + CUDA runtime)
create_env.sh                    One-command environment bootstrap script
```

---

## Installation

This repository provides a one-command Conda setup for macOS and Linux.

### Prerequisites
- Anaconda or Miniconda installed.
- `conda` available in your terminal `PATH`.

### 1) Create the default environment
```bash
bash create_env.sh
```

This creates an environment named `path_foundation` and:
- applies `environment.base.yml` on all platforms;
- applies `environment.mac.yml` on macOS;
- applies `environment.linux-cuda.yml` on Linux when NVIDIA GPU is detected.

### 2) Optional: choose a custom environment name
```bash
bash create_env.sh --name my_env_name
```

### 3) Optional: install Trident/stain-normalisation extras (Linux)
```bash
bash create_env.sh --with-trident
```

This additionally installs `cupy-cuda12x`, `cucim`, `torch-staintools`, and TRIDENT.

### 4) Activate the environment
```bash
conda activate path_foundation
```

---

## End-to-End Quick Start

Minimal reproducible workflow:

### 1) Split case metadata
```bash
python data_split.py \
  --source_csv /path/to/cases.csv \
  --train_size 60 \
  --val_size 20 \
  --test_size 20 \
  --output_folder /path/to/splits
```

### 2) Extract embeddings
Option A: Google Path Foundation (`extract_embedding.py`)
```bash
python extract_embedding.py \
  --wsi_path /path/to/wsi_folder \
  --tile_path /path/to/h5_tiles \
  --model_path /path/to/path_foundation_model \
  --output_path /path/to/output_embeddings
```

Option B: Trident with stain normalisation (`trident_with_normalisation.py`)
```bash
python trident_with_normalisation.py \
  --wsi_dir /path/to/wsis \
  --coords_dir /path/to/coords \
  --out_dir /path/to/output_h5 \
  --target_img_path /path/to/target_image.tif \
  --encoder_name uni_v2 \
  --norm_method macenko \
  --batch_size 128 \
  --gpu 0 \
  --mag 20
```

### 3) Train MIL model
```bash
python train.py \
  --train_folder /path/to/train_embeddings \
  --train_labels /path/to/train_labels.csv \
  --val_folder /path/to/val_embeddings \
  --val_labels /path/to/val_labels.csv \
  --model_folder /path/to/model_output \
  --k_instances 500 \
  --epochs 200 \
  --lr 0.0005 \
  --patience 10 \
  --num_class 14 \
  --emb_type h5
```

### 4) Evaluate MIL model
```bash
python test.py \
  --test_folder /path/to/test_embeddings \
  --test_labels /path/to/test_labels.csv \
  --model /path/to/model_output/mil_best_model_state_dict_epoch_x.pth \
  --output /path/to/test_output \
  --emb_type h5 \
  --cohort cohort_1 \
  --num_class 14
```

### 5) Run optional analyses
```bash
python analysis_scripts/bootstrap.py \
  --csv_path /path/to/test_output/cohort_1/cohort_1_individual_results.csv \
  --n_bootstrap 1000
```

---

## Root Scripts and CLI Usage

### 1) `data_split.py`
Perform a train/validation/test split on a CSV containing `case_id` and `ground_truth`.

CLI arguments:
- `--source_csv` (str, required): input CSV with `case_id` and `ground_truth`.
- `--train_size` (int, default `60`): training split percentage.
- `--val_size` (int, default `20`): validation split percentage.
- `--test_size` (int, default `20`): test split percentage.
- `--output_folder` (str, required): output directory for split CSV files.

---

### 2) `extract_embedding.py`
Extract tile-level embeddings using Google Path Foundation (TensorFlow).

CLI arguments:
- `--wsi_path` (str, required): directory containing source WSIs.
- `--tile_path` (str, required): directory containing CLAM tile coordinate `.h5` files.
- `--model_path` (str, required): Path Foundation model path.
- `--output_path` (str, required): output directory for per-slide embedding CSVs.

---

### 3) `trident_with_normalisation.py`
Optional pipeline that applies stain normalisation before Trident embedding extraction.

CLI arguments:
- `--wsi_dir` (str, required): directory containing WSIs.
- `--coords_dir` (str, required): directory containing `*_patches.h5` coordinate files.
- `--out_dir` (str, required): output directory for generated feature `.h5` files.
- `--target_img_path` (str, required): target image used to fit the normaliser.
- `--encoder_name` (str, default `uni_v2`): Trident encoder name.
- `--norm_method` (str, default `macenko`): one of `vahadane`, `macenko`, `reinhard`.
- `--batch_size` (int, default `128`): patch batch size.
- `--gpu` (int, default `0`): CUDA device index.
- `--mag` (int, default `20`): desired magnification.
- `--custom_list_of_wsis` (str, optional): CSV with header `wsi` to filter processed slides.

Environment setup for this script:
1. Install `torch-staintools` and `cupy`:
```bash
pip install torch-staintools
pip install cupy-cuda12x
```
2. Make `LD_LIBRARY_PATH` persistent for CuPy:
```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/env_nvrtc.sh <<'SH'
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(python - <<'PY'
import os, nvidia.cuda_nvrtc as m
print(os.path.join(os.path.dirname(m.__file__), "lib"))
PY
)"
SH
```
3. Install cuCIM:
```bash
pip install cucim
```

---

### 4) `train.py`
Train an attention-based Multi-Instance Learning (MIL) classifier.

Key behaviour:
- Multiple training folders are supported (`--train_folder`, `--train_folder_2`, `--train_folder_3`).
- Validation runs after each epoch.
- The best model is selected by **lowest validation loss**.
- Early stopping is applied using `--patience`.
- Supports both `h5` and `csv` embeddings.

CLI arguments:
- `--train_folder` (str, required): primary training embedding folder.
- `--train_folder_2` (str, optional): second training embedding folder.
- `--train_folder_3` (str, optional): third training embedding folder.
- `--train_labels` (str, required): training label CSV.
- `--val_folder` (str, required): validation embedding folder.
- `--val_labels` (str, required): validation label CSV.
- `--model_folder` (str, required): output folder for checkpoints and logs.
- `--k_instances` (int, default `500`): number of instances sampled per slide.
- `--epochs` (int, default `200`): number of epochs.
- `--lr` (float, default `0.0005`): learning rate.
- `--patience` (int, default `10`): early-stopping patience.
- `--num_class` (int, default `14`): number of classes.
- `--emb_type` (str, default `h5`): `h5` or `csv`.

---

### 5) `test.py`
Evaluate a trained MIL classifier on a held-out test set.

CLI arguments:
- `--test_folder` (str, required): test embedding folder.
- `--test_labels` (str, required): test label CSV with `case_id` and `ground_truth`.
- `--model` (str, required): path to model checkpoint.
- `--output` (str, required): output folder for metrics and predictions.
- `--emb_type` (str, default `h5`): `h5` or `csv`.
- `--cohort` (str, required): cohort identifier used as output subfolder name.
- `--num_class` (int, default `14`): number of classes.

Example:
```bash
python test.py \
  --test_folder /path/to/test_embeddings \
  --test_labels /path/to/test_labels.csv \
  --model /path/to/mil_best_model_state_dict_epoch_x.pth \
  --output /path/to/output_predictions \
  --emb_type h5 \
  --cohort cohort_1 \
  --num_class 14
```

---

### 6) `run_logistic_regression.py`
Run slide-level Logistic Regression inference on precomputed slide embeddings.

Expected metadata format:
- `case_id`
- `ground_truth` (1-based class index)
- `split` (`train`, `val`, or `test`)

CLI arguments:
- `--metadata_file` (str, required): metadata CSV in the format above.
- `--embedding_dir` (str, required): folder containing `<case_id>.h5` slide-level embeddings.

Output files (written to the current working directory):
- `accuracy_metrics.csv`
- `classwise_accuracy.csv`
- `top3_predictions.csv`

Example:
```bash
python run_logistic_regression.py \
  --metadata_file /path/to/metadata_with_split.csv \
  --embedding_dir /path/to/slide_level_embeddings
```

---

## Troubleshooting

- **`openslide` import or runtime errors**
  Install OpenSlide system libraries and Python bindings together. On many Linux systems you need both the OS package and `pip install openslide-python`.

- **CuPy/CUDA mismatch in `trident_with_normalisation.py`**
  Ensure your CuPy build matches your CUDA runtime (for example `cupy-cuda12x` for CUDA 12.x). Mismatched versions usually fail at import or first GPU call.

- **Missing datasets in `.h5` files**
  Training/testing utilities expect valid embedding files. In particular, many workflows expect `.h5` files to include `features` (and, for some scripts, `coords`).

---

## Licence and Data Usage

### Code licence
This repository is licensed under the **Apache License 2.0**. See `LICENSE`.

### Dataset licence (checked from linked pages)
- The dataset portal page (`bhchai.com/visualise_scan_stain_efffects/dataset.html`) lists Kaggle dataset links with licence of **Attribution 4.0 International (CC BY 4.0)**.

Please always verify the latest licence terms on the source hosting platform before redistribution or commercial use.

### Data usage and citation
- Follow any data access conditions on Kaggle and the source pages.
- If you use this dataset or code in research, please cite the paper below.

---

## Useful Links
- **Trident**: https://github.com/mahmoodlab/TRIDENT/
- **cuCIM**: https://github.com/rapidsai/cucim
- **torch-staintools**: https://github.com/CielAl/torch-staintools

---

## Citation
If you use this repository, code or the published dataset in your research, please cite:
```bibtex
@article{chai2025impact,
  title={Impact of tissue staining and scanner variation on the performance of pathology foundation models: a study of sarcomas and their mimics},
  author={Chai, Binghao and Chen, Jianan and Cool, Paul and Oumlil, Fatine and Tollitt, Anna and Steiner, David F and Chakraborti, Tapabrata and Flanagan, Adrienne M},
  journal={bioRxiv},
  pages={2025--08},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```
