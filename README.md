# Rare Disease Diagnosis from Clinical Text

Exploring embedding-based approaches for diagnosing rare diseases from clinical case summaries. The project compares classical NLP baselines against biomedical embedding models for both **classification** and **retrieval** tasks.

## Dataset

- **6,915 clinical case summaries** mapped to **1,012 rare disease diagnoses**
- Highly imbalanced long-tail distribution: 282 diseases have only 1 sample, only 15 have 50+
- Two evaluation splits:
  - **Classification split**: 320 classes with 6+ samples (4,312 train / 1,079 test)
  - **Retrieval split**: all 1,012 classes (5,532 train / 1,383 test)

## Notebooks

| # | Notebook | Task | Approach | Key Result |
|---|---|---|---|---|
| 00 | `00_eda.ipynb` | EDA | Dataset exploration, text statistics, class distribution | 137K unique tokens, severe class imbalance |
| 01 | `01_baseline_TF_IDF.ipynb` | Classification | TF-IDF + Logistic Regression | Accuracy: 0.316, F1-macro: 0.291 |
| 02 | `02_biolord_xgboost_fulltext.ipynb` | Classification | BioLORD-2023 (flat) + XGBoost | Accuracy: 0.180, F1-macro: 0.111 |
| 03 | `03_biolord_chunked_embs_xgboost.ipynb` | Classification | BioLORD-2023 (chunked mean_max) + XGBoost | Accuracy: 0.243, F1-macro: 0.166 |
| 04 | `04_differential_diagnosis.ipynb` | Retrieval | BioLORD-2023 + k-NN cosine | Hit@10: 0.430, MRR: 0.227 |
| 05 | `05_retrieval_embeddings.ipynb` | Retrieval | MedCPT vs BioLORD (3 variants) | Hit@10: 0.470, MRR: 0.269 |

## Key Findings

**Classification:** TF-IDF + Logistic Regression outperforms dense embeddings. Mean-pooled embeddings dilute disease-discriminating tokens that TF-IDF preserves.

**Chunking consistently helps:** Splitting clinical text into 150-word chunks with mean_max aggregation improves both classification (+6pp accuracy) and retrieval (+4-5pp Hit@10) across all embedding models.

**MedCPT > BioLORD for retrieval:** A contrastive retrieval model (MedCPT) outperforms a semantic similarity model (BioLORD) by ~4pp across all metrics. However, the asymmetric query/article setup fails because clinical descriptions are not short search queries.

**Absolute retrieval performance remains moderate:** Best Hit@10 = 47%. The cosine similarity gap between correct and incorrect retrievals is only 0.006, indicating that the embedding space lacks disease-level discriminability.

**Class frequency drives performance:** Hit@10 ranges from 13% for diseases with 1 training sample to 68% for diseases with 21+ samples.

## Project Structure

```
rare_disease_diagnosis/
├── notebooks/          # Experiment notebooks (00-05)
├── src/
│   ├── embeddings.py   # BioLORD and MedCPT embedding generation
│   └── chunking.py     # Text chunking and chunk-to-document aggregation
├── data/               # Dataset, splits, and pre-computed embeddings (not tracked)
└── requirements.txt
```

## Models Used

- **[BioLORD-2023](https://huggingface.co/FremyCompany/BioLORD-2023)** — Biomedical sentence embeddings trained for semantic similarity (768-dim)
- **[MedCPT](https://huggingface.co/ncbi/MedCPT-Article-Encoder)** — Contrastive pre-trained model for medical information retrieval (768-dim, separate query/article encoders)

## Setup

```bash
git clone https://github.com/amlopeza/rare_disease_diagnosis.git
cd rare_disease_diagnosis
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Additional dependencies for embedding generation (run on GPU/Colab):
```bash
pip install sentence-transformers transformers torch xgboost nltk wordcloud
```

## Next Steps

- **Contrastive fine-tuning** with triplet or supervised contrastive loss (same diagnosis = positive pair) to improve embedding space discriminability
- **LLM-based re-ranking** over top-k retrieved candidates

## Disclaimer

This project is for research and educational purposes only.  
It is not intended for clinical use, medical decision-making, or real-world diagnostic applications.

The dataset used in this project is licensed under CC BY 4.0 and sourced from ZebraMap (via Kaggle). Proper attribution is provided in accordance with the dataset license.
