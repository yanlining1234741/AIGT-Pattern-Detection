# Dataset Download & Access Instructions

This project uses five publicly available datasets. All datasets are free to access.  
**None of the raw dataset files are included in this repository** due to file size and licensing constraints.

After downloading, place all files in the `datasets/raw/` directory.

---

## Dataset 1 — DAIGT v2 (Detect AI Generated Text)

**Used as:** `DAIGT_Clean_Balanced.csv`  
**Size:** ~137k samples | **Domain:** Student essays, general text  
**Labels:** Human (0) vs. multi-LLM generated (1)

### Download

1. Go to: https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset
2. Sign in to Kaggle (free account required)
3. Click **Download** → extract the zip
4. Rename the main CSV to `train_essays_v2.csv`
5. Place in `datasets/raw/`

**Or via Kaggle CLI:**
```bash
pip install kaggle
kaggle datasets download -d thedrcat/daigt-v2-train-dataset
unzip daigt-v2-train-dataset.zip -d datasets/raw/
```

**Expected columns:** `text`, `generated`, `prompt_name`

---

## Dataset 2 — gsingh1-py (MultiModel — Human vs. AI News)

**Used as:** `MultiModel_Clean_Balanced.csv`  
**Size:** ~7,320 samples | **Domain:** News articles  
**Labels:** Human (0) vs. multiple LLMs including Gemma, Mistral, GPT-4o (1)

### Download

1. Go to: https://huggingface.co/datasets/gsingh1-py/train
2. Click **Files and versions** tab
3. Download the Parquet or CSV file
4. Save as `gsingh1_py_train.csv` in `datasets/raw/`

**Or via Python:**
```python
from datasets import load_dataset
ds = load_dataset("gsingh1-py/train")
ds["train"].to_csv("datasets/raw/gsingh1_py_train.csv", index=False)
```

**Expected columns:** `text`, `label` (or similar; preprocessing handles column mapping)

---

## Dataset 3 — HC3 (Human ChatGPT Comparison Corpus)

**Used as:** `HC3_Clean_Balanced.csv`  
**Size:** ~24k Q&A pairs | **Domain:** Finance, medicine, law, open QA  
**Labels:** Human expert answers (0) vs. ChatGPT answers (1)

### Download

1. Go to: https://github.com/Hello-SimpleAI/chatgpt-comparison-detection
2. Follow the instructions in the repo's README to download the merged CSV
3. Save as `HC3_all_merged.csv` in `datasets/raw/`

**Or direct HuggingFace download:**
```python
from datasets import load_dataset
ds = load_dataset("Hello-SimpleAI/HC3", name="all")
import pandas as pd
df = ds["train"].to_pandas()
df.to_csv("datasets/raw/HC3_all_merged.csv", index=False)
```

**Expected columns:** `question`, `human_answers`, `chatgpt_answers`, `source`, `subset`

---

## Dataset 4 — LLMDetect (Kaggle Competition)

**Used as:** `LLMDetect_Clean_Balanced.csv`  
**Size:** ~1,375 samples | **Domain:** Student argumentative essays  
**Labels:** Human student (0) vs. LLM-generated (1) — strongly human-imbalanced

### Download

1. Go to: https://www.kaggle.com/competitions/llm-detect-ai-generated-text/data
2. Accept the competition rules (free)
3. Download `train_essays.csv`
4. Rename to `LLM_Detect AI_train_essays.csv` and place in `datasets/raw/`

**Or via Kaggle CLI:**
```bash
kaggle competitions download -c llm-detect-ai-generated-text
unzip llm-detect-ai-generated-text.zip -d datasets/raw/
```

**Expected columns:** `id`, `text`, `generated`

---

## Dataset 5 — MAGE (Machine-Generated Text Detection in the Wild)

**Used as:** `MAGE_test_Clean_Balanced.csv`  
**Size:** ~50k samples | **Domain:** 10+ domains, 27 LLMs, real-world attacks  
**Labels:** Human (0) vs. AI-generated (1)

### Download

```python
from datasets import load_dataset
ds = load_dataset("yaful/MAGE")
ds["test"].to_pandas().to_csv("datasets/raw/MAGE_test.csv", index=False)
```

**Or via HuggingFace Hub:**
1. Go to: https://huggingface.co/datasets/yaful/MAGE
2. Click **Files and versions** → download the test split parquet file
3. Convert to CSV and save as `MAGE_test.csv` in `datasets/raw/`

**Expected columns:** `text`, `label`, `src`

---

## Verifying Your Setup

After downloading all datasets, your `datasets/raw/` folder should contain:

```
datasets/raw/
├── train_essays_v2.csv                  # DAIGT
├── gsingh1_py_train.csv                 # MultiModel
├── HC3_all_merged.csv                   # HC3
├── LLM_Detect AI_train_essays.csv       # LLMDetect
└── MAGE_test.csv                        # MAGE
```

You can verify with:
```bash
ls datasets/raw/
```

Then proceed to run `notebooks/01_preprocessing.ipynb`.

---

## Storage Requirements

| Dataset | Raw Size (approx.) |
|---------|--------------------|
| DAIGT | ~150 MB |
| MultiModel | ~15 MB |
| HC3 | ~50 MB |
| LLMDetect | ~5 MB |
| MAGE | ~200 MB |
| **Total raw** | **~420 MB** |
| Processed + features | ~600 MB additional |

**Recommended free disk space: at least 2 GB**

---

## Notes

- All datasets are publicly available and free to use for academic research.
- The Kaggle datasets require a free Kaggle account.
- HuggingFace datasets require the `datasets` library (`pip install datasets`).
- Preprocessing handles all column name variations automatically — do not rename columns inside the CSV files.
