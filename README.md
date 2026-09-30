# Leakage-Aware and Subject-Level Evaluation of DINOv2 Knowledge Distillation for CDR-Based Dementia Severity Classification from MRI Slices

Code, split files, hashes, per-item predictions, statistics and figures for the article by **Ganesan M** and **Lakshmanan M**
(Dayananda Sagar University). The study is a leakage-aware and subject-level evaluation of CDR-based dementia-severity
classification from MRI slices; knowledge distillation (KD) from a frozen DINOv2 ViT-B/14 teacher to a DINOv2 ViT-S/14 student
(logit + patch-feature transfer) is the methodological case study.

**Main findings**
- The widely used Kaggle four-class archive stores its test folder a second time as its training folder (6,400 of 12,800 files are unique), and its slices trace back to OASIS-1 subjects (32 slices per subject; class = CDR).
- One fixed classifier scores 99.99 % macro-F1 on the duplicated folder, 96.3 % when oversampling precedes the split and 86.9 % under a clean split.
- Under subject-level five-fold validation on OASIS-1 (347 subjects) the distilled student reaches 56.4 % subject-level macro-F1.
- The pre-specified hypothesis that KD improves the student was not supported (cross-validated macro-F1 88.39 % vs 88.86 %).

---

## 1. Quick start for reviewers — reproduce every reported number in minutes (no data, no GPU)

All models were trained once and **every prediction is stored** in `results/predictions/`. The statistics, tables and figures
of the paper are recomputed from these predictions:

```bash
git clone https://github.com/YOUR-GITHUB-USERNAME/leakage-aware-dinov2-kd.git
cd leakage-aware-dinov2-kd
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/reproduce_results.py --quick              # < 2 min: tables, figures, headline checks
python scripts/reproduce_results.py                      # about 20 min: also recomputes all bootstrap / permutation statistics
```

Output:
- `results/metrics/TABLES.md` — every results table of the paper (also as `results/metrics/table_*.csv`)
- `figures/generated/` — Figs. 19–24 at 900 dpi
- a list of **headline checks** (`PASS` / `FAIL`) that compares the recomputed values with the values printed in the manuscript
  (`results/metrics/expected_headline_values.json`), ending with `ALL CHECKS PASSED`.

## 2. Reproduce the leakage audit on the original archive (1 minute, needs only the Kaggle download)

```bash
# download https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images
# and unzip it so that data/kaggle/Alzheimer_s Dataset/train and .../test exist (see docs/DATA_ACCESS.md)
python scripts/leakage_audit.py
```
The script hashes all 12,800 files (MD5 + SHA-256), finds the 6,400 duplicate pairs and checks that your copy matches the
released hashes (`hashes/`, `splits/kaggle_unique_slices.csv`).

## 3. Re-run the experiments from the raw data (CPU, no GPU required)

```bash
python scripts/download_dinov2_weights.py     # official DINOv2 ViT-S/14 and ViT-B/14 checkpoints -> weights/
python scripts/check_environment.py          # shows which inputs are present
python scripts/smoke_test.py                 # 2-minute test of the whole training stack
bash   scripts/run_full_pipeline.sh          # every experiment in the paper, in order (about 40 CPU-hours on 2 threads)
```
Each experiment writes an append-only JSONL log (one line per trained model, with its test predictions) and **resumes automatically**
if it is interrupted. The pipeline ends with `scripts/reproduce_results.py`. Environment variables `DATA_DIR`, `DINO_S`, `DINO_B`
override the default locations `data/`, `weights/dinov2_vits14_pretrain.pth`, `weights/dinov2_vitb14_pretrain.pth`.
Deep-learning results can differ in the last digit between CPUs and library versions; the reported numbers are those in `results/predictions/`.

## 4. Repository structure

```
leakage-aware-dinov2-kd/
├── README.md, LICENSE (MIT), CITATION.cff, requirements.txt, environment.yml
├── configs/                  all hyper-parameters, seeds and protocols (data, student_kd, student_ce, revision_experiments)
├── src/                      metrics (accuracy, macro-F1, MCC, κ, per-class), McNemar, Grad-CAM, tuners
├── experiments/cpu/          the complete experiment pipeline (see docs/REPRODUCIBILITY_MAP.md)
├── stats/                    statistics: bootstrap CIs, permutation/McNemar tests, Holm, effect sizes, ANOVA, ordinal, calibration
├── scripts/                  reproduce_results.py, generate_tables.py, generate_figures.py, leakage_audit.py,
│                             run_full_pipeline.sh, smoke_test.py, check_environment.py, download_dinov2_weights.py
├── splits/                   Kaggle unique files (hashes, hold-out split, CV fold); OASIS-1 slices, subject folds, known-young subjects
├── hashes/                   MD5/SHA-256 of all 12,800 archive files and the duplicate-audit summary
├── provenance/               Kaggle ↔ OASIS-1 matching results
├── results/
│   ├── predictions/          JSONL run logs with per-item predictions/probabilities of every model and run (+ journal/ subfolder)
│   ├── metrics/              regenerated tables (TABLES.md, table_*.csv) and expected headline values
│   ├── statistical_tests/    copies of the statistics outputs (tests, CIs, effect sizes)
│   └── logs/                 console logs of the original runs
├── figures/paper/            all figures of the article (Figs. 1–24)
├── models/                   p1_final_seed0.pt — trained DistilDINO-AD student (blocks 9–12 + head), seed 0
└── docs/                     DATA_ACCESS.md, REPRODUCIBILITY_MAP.md, RESULTS_FILES.md
```

## 5. Data (not redistributed)

The MRI data and the DINOv2 weights are **not** included. Obtain them from the original sources (details in `docs/DATA_ACCESS.md`):
- Kaggle “Alzheimer’s dataset (4 class of images)” (tourist55).
- OASIS-1 (https://sites.wustl.edu/oasisbrains/), under the OASIS data use agreement: the OASIS-1 slice collection and, for the age
  and provenance analyses, the OASIS-1 disc archives 1–4.
- DINOv2 checkpoints from https://github.com/facebookresearch/dinov2 (downloaded by `scripts/download_dinov2_weights.py`).

The repository contains only derived, non-image material: file hashes, file and subject lists, labels, split assignments and
model predictions.

## 6. Hardware and software used for the paper

Intel Xeon @ 2.80 GHz, **2 CPU threads**, 7.8 GB RAM, no GPU; Linux; Python 3.11.15, PyTorch 2.14.0, timm 1.0.30.
One four-block student run (8 epochs) takes about 9–13 minutes, an eight-block run about 20–30 minutes.

## 7. Citation

See `CITATION.cff`. Archived release: Zenodo DOI **10.5281/zenodo.XXXXXXX** (added after the first release).

## 8. License

Code: MIT (see `LICENSE`). Data and pretrained weights remain under their own licences and terms of use.
