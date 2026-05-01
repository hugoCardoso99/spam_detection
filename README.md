# Spam Detection — Model Comparison

A practice project comparing **Logistic Regression**, **Random Forest**, and **XGBoost** on the [SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) dataset.

## Project Structure

```
spam_detection/
├── data/                   # Raw & preprocessed data (auto-downloaded)
├── models/                 # Saved models + TF-IDF vectorizer
├── notebooks/              # (optional) Exploratory analysis
├── src/
│   ├── preprocess.py       # Download, clean, TF-IDF, train/test split
│   ├── train.py            # Train all three models
│   ├── evaluate.py         # Compare models on test set
│   └── inference.py        # Predict on new messages
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt

# 1. Download data & build features
python src/preprocess.py

# 2. Train all models
python src/train.py

# 3. Evaluate & compare
python src/evaluate.py

# 4. Predict on new messages
python src/inference.py "Congratulations! You won a free prize"
python src/inference.py "Hey, are we meeting at 5 today?"
```

## Pipeline Overview

1. **Preprocessing** — download the SMS Spam Collection, lowercase text, strip URLs/punctuation/digits, then vectorise with TF-IDF (unigrams + bigrams, top 5 000 features).
2. **Training** — fit Logistic Regression, Random Forest (200 trees), and XGBoost (200 rounds) on the 80 % training split.
3. **Evaluation** — compare all models on the 20 % held-out test set using accuracy, precision, recall, F1-score, ROC-AUC, and **PR-AUC** (critical for imbalanced classes).
4. **Inference** — load the best model and predict spam/ham on arbitrary text.

## Model Comparison Results

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---------------------|----------|-----------|--------|----------|---------|--------|
| Logistic Regression | 0.9623   | 1.0000    | 0.7181 | 0.8359   | 0.9865  | 0.9645 |
| **Random Forest**   | **0.9794** | **1.0000** | **0.8456** | **0.9164** | **0.9849** | **0.9627** |
| XGBoost             | 0.9623   | 0.9280    | 0.7785 | 0.8467   | 0.9630  | 0.9076 |

## Which Model Is Best — and Why?

**Random Forest** is the best-performing model for this task, selected primarily on **PR-AUC** — the most appropriate metric for imbalanced classification.

### Why PR-AUC is the primary metric

Spam detection is a **class-imbalanced** problem (~87 % ham, ~13 % spam). This has consequences for metric selection:

- **Accuracy** is misleading — a model that predicts "ham" for everything still scores ~87 %.
- **ROC-AUC** is inflated by the large number of true negatives. All three models score between 0.96–0.99, making it hard to differentiate them. ROC-AUC measures the trade-off between true positive rate and false positive rate, but when negatives massively outnumber positives, even a poor model rarely produces false positives relative to the total negative count.
- **PR-AUC** (`average_precision_score`) focuses entirely on the positive (spam) class. It measures how well the model maintains high precision as recall increases, without being propped up by easy negative predictions. This is where meaningful model differences actually surface.

**F1-score** is also important as it captures the precision–recall trade-off at a single threshold, but PR-AUC evaluates performance across all thresholds, giving a more complete picture.

### Why Random Forest wins

Random Forest achieves the best scores on almost every metric that matters:

- **PR-AUC: 0.9627** — highest of the three, meaning it maintains the best precision–recall balance across all classification thresholds.
- **F1-score: 0.9164** — significantly ahead of both Logistic Regression (0.8359) and XGBoost (0.8467), driven by perfect precision and the highest recall.
- **Precision: 1.0000** — zero false positives. When Random Forest flags a message as spam, it is always correct. No legitimate messages end up in the spam folder.
- **Recall: 0.8456** — highest of all three models. It catches more actual spam than either competitor.
- **Accuracy: 0.9794** — the highest overall correctness rate.

Random Forest dominates because it achieves the best trade-off: it never misclassifies a ham message as spam (perfect precision), while simultaneously catching the most spam (highest recall). That combination is exactly what PR-AUC rewards.

### Why not XGBoost?

XGBoost underperforms here despite being a more complex model. Its precision drops to 0.928 (it produces false positives that the other two avoid), its recall is lower than Random Forest's, and its PR-AUC of 0.9076 is the worst of the three. Gradient boosting doesn't always outperform simpler ensembles — on smaller, well-structured TF-IDF feature spaces, Random Forest's bagging approach can generalise better than boosting, which is more prone to overfitting.

### Why not Logistic Regression?

Logistic Regression matches Random Forest on precision (perfect 1.0) but falls behind significantly on recall (0.7181 vs. 0.8456). It misses roughly 28 % of spam. As a linear model, it cannot capture the non-linear interactions between TF-IDF features that the tree-based ensembles leverage. That said, it remains the simplest and fastest option, and its perfect precision makes it a reasonable choice if minimising false positives is the only priority.

### Summary

| Criterion             | Winner              | Reason                                      |
|-----------------------|---------------------|---------------------------------------------|
| Best PR-AUC           | Random Forest       | Best precision–recall trade-off across all thresholds |
| Best F1-score         | Random Forest       | Highest harmonic mean of precision & recall  |
| Best precision        | LR / Random Forest  | Both achieve perfect 1.0 — zero false positives |
| Best recall           | Random Forest       | Catches the most spam (84.6 %)               |
| Fastest / simplest    | Logistic Regression | Linear model, trains in milliseconds         |
| Best overall          | **Random Forest**   | Top PR-AUC + top F1 + perfect precision + best recall |

## License

MIT — use freely for learning and interviews.
