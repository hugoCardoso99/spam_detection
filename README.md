# Spam Detection — Model Comparison

A practice project comparing **Logistic Regression**, **Random Forest**, and **XGBoost** on spam/ham text classification datasets.

Works with any CSV that has a `Message` column and a `Spam/Ham` column. Includes built-in support for the [SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) dataset.

## Project Structure

```
spam_detection/
├── data/<dataset>/         # Preprocessed splits (per dataset)
├── figures/<dataset>/      # Evaluation plots (per dataset)
├── models/<dataset>/       # Saved models + TF-IDF vectorizer (per dataset)
├── notebooks/
│   └── model_comparison.ipynb  # Full analysis, plots, and model comparison
├── src/
│   ├── preprocess.py       # Load, clean, TF-IDF, train/test split
│   ├── train.py            # Train all three models
│   ├── evaluate.py         # Compare models + generate figures
│   └── inference.py        # Predict on new text
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
```

### SMS Spam Collection (built-in)

```bash
python src/preprocess.py --dataset sms
python src/train.py --dataset sms
python src/evaluate.py --dataset sms
python src/inference.py --dataset sms "You won a free prize!"
```

### Custom CSV dataset

Any CSV with columns `Message` and `Spam/Ham` (case-insensitive):

```bash
python src/preprocess.py --dataset enron --csv data/enron_labeled.csv
python src/train.py --dataset enron
python src/evaluate.py --dataset enron
python src/inference.py --dataset enron "urgent wire transfer needed"
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
