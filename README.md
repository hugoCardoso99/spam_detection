# Spam Detection — Model Comparison

A practice project comparing **Logistic Regression**, **Random Forest**, and **XGBoost** on spam/ham text classification datasets.

Works with any CSV that has a `Message` column and a `Spam/Ham` column. Includes built-in support for the [SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) dataset.

## Project Structure

```
spam_detection/
├── data/<dataset>/         # Preprocessed splits (per dataset, optional)
├── models/<dataset>/       # Saved models + TF-IDF vectorizer (per dataset)
├── notebooks/              # Analysis notebooks
├── src/
│   ├── preprocess.py       # Load, clean, TF-IDF, train/test split
│   ├── train.py            # Train all three models
│   ├── evaluate.py         # Compare models + generate figures
│   ├── inference.py        # Predict on new text
│   └── db.py               # PostgreSQL persistence layer
├── docker-compose.yml      # Postgres container config
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
```

### File-based pipeline (default)

```bash
python src/preprocess.py --dataset sms
python src/train.py --dataset sms
python src/evaluate.py --dataset sms
python src/inference.py --dataset sms "You won a free prize!"
```

### PostgreSQL-backed pipeline (scalable)

For larger datasets, raw messages are persisted in Postgres while models and vectorizers remain as joblib files.

**1. Start Postgres (via Docker Compose):**

```bash
docker compose up -d
```

**2. Create `.env` from the template:**

```bash
cp .env.example .env
```

**3. Run the pipeline with `--use-db` / `--from-db` flags:**

```bash
python src/preprocess.py --dataset sms --use-db
python src/train.py --dataset sms --from-db
python src/evaluate.py --dataset sms --from-db
python src/inference.py --dataset sms "You won a free prize!"
```

Or with a custom CSV:

```bash
python src/preprocess.py --dataset enron --csv data/enron_labeled.csv --use-db
python src/train.py --dataset enron --from-db
python src/evaluate.py --dataset enron --from-db
```

## Pipeline Overview

1. **Preprocessing** — load a CSV (or auto-download SMS Spam Collection), clean text, vectorise with TF-IDF (unigrams + bigrams, top 5,000 features), and split 80/20.
2. **Training** — fit Logistic Regression, Random Forest (200 trees), and XGBoost (200 rounds).
3. **Evaluation** — compare models using accuracy, precision, recall, F1-score, ROC-AUC, and PR-AUC. Generate confusion matrices (at optimal F1 threshold), PR curves, and F1-vs-threshold plots.
4. **Inference** — load a trained model and classify new text as spam or ham.

All artefacts are namespaced by `--dataset`, so multiple datasets can coexist without overwriting each other.

## Model Comparison

For the full analysis — metric tables, visualisations, and interpretation — see:

**[notebooks/model_comparison.ipynb](notebooks/model_comparison.ipynb)**

## License

MIT — use freely for learning and interviews.
