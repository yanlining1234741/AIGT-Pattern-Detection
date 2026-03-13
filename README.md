# AIGT-Pattern-Detection

## Adaptive AI-Generated Text Detection via Generative Pattern of LLM

**Author:** Lining Yan  
**Supervisor:** Dr. Baha Ihnaini  
**Institution:** Wenzhou-Kean University, Department of Computer Science and Technology  
**Year:** 2025

---

## Overview

This repository contains the full implementation for the bachelor's thesis:  
**"Adaptive AI-Generated Text Detection via Generative Pattern of LLM"**

The project proposes a **pattern-centric, data-efficient, and interpretable** framework for detecting AI-generated text (AIGT). Rather than relying on large end-to-end neural models, the system explicitly extracts reusable **generative patterns** from linguistic feature space and uses them for downstream classification.

### Key Contributions

- Multi-granularity feature extraction (statistical, syntactic, semantic)
- Clustering-based generative pattern discovery (KMeans & DBSCAN)
- Pattern-aware low-resource training (44–50 labeled samples)
- Interpretable classification using EBM and XGBoost with SHAP analysis
- Transformer baselines (BERT, RoBERTa) for comparison

---

## Project Structure

```
AIGT-Pattern-Detection/
│
├── notebooks/
│   ├── 01_preprocessing.ipynb       # Dataset loading, cleaning, balancing (5 datasets)
│   ├── 02_feature_extraction.ipynb  # Feature extraction + pattern discovery (KMeans/DBSCAN)
│   └── 03_model_training.ipynb      # EBM, XGBoost, BERT, RoBERTa training & evaluation
│
├── datasets/
│   ├── raw/                         # Place raw downloaded datasets here (see DATASETS.md)
│   ├── processed/                   # Output of 01_preprocessing.ipynb
│   ├── features/                    # Output of 02_feature_extraction.ipynb
│   │   └── pattern_training_sets/   # Pattern-sampled training sets
│   └── splits/                      # Train/test splits (5k each)
│
├── results/                         # All output CSVs and plots
│
├── docs/
│   └── DATASETS.md                  # Dataset download & access instructions
│
├── requirements.txt                 # All Python dependencies
├── .gitignore
└── README.md
```

---

## Environment Setup

### Requirements

- Python 3.9 or higher
- CUDA-capable GPU recommended (tested on CUDA 12.1); CPU mode also supported

### Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/AIGT-Pattern-Detection.git
cd AIGT-Pattern-Detection
```

### Step 2 — Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Install spaCy Language Model

```bash
python -m spacy download en_core_web_sm
```

### Step 5 — (Optional) Configure Hugging Face Cache Directory

If you have limited space on your system drive, set the cache to another location before running notebooks:

```bash
# Windows
set HF_HOME=D:\huggingface_cache

# macOS / Linux
export HF_HOME=/your/preferred/cache/path
```

---

## Dataset Setup

See [`docs/DATASETS.md`](docs/DATASETS.md) for detailed download instructions for all 5 datasets.

Once downloaded, place the raw files in `datasets/raw/` with these exact filenames:

| File | Dataset |
|------|---------|
| `train_essays_v2.csv` | DAIGT v2 |
| `gsingh1_py_train.csv` | gsingh1-py (MultiModel) |
| `HC3_all_merged.csv` | HC3 |
| `LLM_Detect AI_train_essays.csv` | LLMDetect |
| `MAGE_test.csv` | MAGE |

---

## Running the Experiments

Run the three notebooks **in order**. Each notebook saves its outputs to disk for the next stage.

### Step 1 — Preprocessing (`01_preprocessing.ipynb`)

**What it does:**
- Loads each raw dataset
- Applies strict text cleaning (URL removal, ASCII whitelist, length filtering)
- Balances classes to 1:1 ratio (undersampling)
- Creates stratified train/test splits (5k samples each, text-disjoint)
- Saves cleaned CSVs to `datasets/processed/` and `datasets/splits/`

**Output files (in `datasets/processed/`):**
```
DAIGT_Clean_Balanced.csv
HC3_Clean_Balanced.csv
LLMDetect_Clean_Balanced.csv
MAGE_test_Clean_Balanced.csv
MultiModel_Clean_Balanced.csv
```

**Update paths at the top of each dataset cell before running:**
```python
CSV_PATH = r"datasets/raw/HC3_all_merged.csv"   # ← change to your path
SAVE_DIR = Path("datasets/processed")
```

---

### Step 2 — Feature Extraction (`02_feature_extraction.ipynb`)

**What it does:**
- Loads the `AdaptiveFeatureExtractor` class which computes:
  - **Statistical:** avg/var sentence length, type-token ratio, punctuation frequency, GPT-2 perplexity
  - **Syntactic:** POS tag ratios (NOUN/VERB/ADJ/ADV/PRON/DET/ADP/CONJ), dependency depth
  - **Semantic:** sentence-level coherence (cosine similarity via `all-MiniLM-L6-v2`)
- Applies KMeans clustering (k=6) and DBSCAN clustering for pattern discovery
- Builds pattern-aware training sets by sampling representative instances per cluster
- Saves feature CSVs and pattern summary to `datasets/features/`

**Models downloaded automatically on first run:**
- `gpt2` (from Hugging Face) — for perplexity
- `all-MiniLM-L6-v2` (from Sentence Transformers) — for semantic coherence

**Update the base directory path at the top of the notebook:**
```python
FEATURE_DIR = Path("datasets/features")
```

---

### Step 3 — Model Training & Evaluation (`03_model_training.ipynb`)

**What it does:**
- Trains and evaluates four model types under full and pattern-only conditions:
  - **EBM** (Explainable Boosting Machine) — full & pattern
  - **XGBoost** — full & pattern  
  - **BERT-base** — pattern-only (fine-tuned, 3 epochs)
  - **RoBERTa-base** — pattern-only (fine-tuned, 3 epochs)
- Ablation studies: feature group contributions (F1/F2/F3/F4), clustering strategies
- Data efficiency sensitivity analysis (varying training size)
- SHAP-based interpretability analysis for EBM
- Saves all result CSVs and plots to `results/`

**Key results produced:**
```
results/xgb_results.csv
results/gbm_results.csv
results/bert_roberta_results.csv
results/full_vs_pattern_auc_EBM.png
results/data_efficiency_EBM.png
results/overall_auc_comparison.png
```

---

## Key Results Summary

| Dataset | EBM-full | EBM-pattern | XGB-full | XGB-pattern | BERT-pattern | RoBERTa-pattern |
|---------|----------|-------------|----------|-------------|--------------|-----------------|
| DAIGT | 0.9953 | 0.9482 | 0.9964 | 0.9563 | 0.9947 | 0.9894 |
| HC3 | 0.9986 | 0.9945 | 0.9991 | 0.9954 | 0.9945 | 0.9949 |
| LLMDetect | 0.8289 | 0.6052 | 0.8628 | 0.6224 | 0.6053 | 0.5729 |
| MAGE | 0.8300 | 0.7075 | 0.8719 | 0.7211 | 0.5704 | 0.6219 |
| MultiModel | 0.7125 | 0.5876 | 0.7068 | 0.5814 | 0.6437 | 0.6240 |

*All pattern-based models trained on only 44–50 samples.*

---

## Dependencies

See `requirements.txt` for the full list. Core packages:

| Package | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data handling |
| `scikit-learn` | Preprocessing, clustering, metrics |
| `xgboost` | XGBoost classifier |
| `interpret` | Explainable Boosting Machine (EBM) |
| `spacy` | Syntactic feature extraction |
| `transformers` | GPT-2 perplexity, BERT/RoBERTa fine-tuning |
| `sentence-transformers` | Semantic coherence (MiniLM) |
| `shap` | Feature attribution analysis |
| `torch` | Deep learning backend |
| `datasets` | HuggingFace dataset utilities |
| `matplotlib`, `seaborn` | Visualisation |

---

## Citation

If you use this code, please cite:

```
Yan, L. (2025). Adaptive AI-Generated Text Detection via Generative Pattern of LLM.
Bachelor's Thesis, Wenzhou-Kean University.
```

---

## License

This project is released for academic and research purposes. Please contact the author for other uses.
