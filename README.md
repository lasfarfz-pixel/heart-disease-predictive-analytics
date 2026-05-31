# Project Analysis — Heart Failure Prediction

## Overview

This document walks through the key findings, decisions, and insights from this project. The goal was to predict whether a patient has heart disease based on clinical data, and to compare the performance of two machine learning approaches: Support Vector Machine (SVM) and XGBoost.

*Dataset:* [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data) — 918 patients, 11 clinical features.

---
## 📁 Project Structure
```text
heart-disease-predictive-analytics/
│
├── data/
│   └── heart.csv                    # Raw clinical patient dataset
│
├── src/
│   ├── data_preprocessing.py        # Automated cleaning, imputation & feature encoding
│   └── models.py                    # Production model training & validation scripts
│
├── heart-failure-prediction.ipynb   # Experimental pipeline & visualization notebook
├── report.pdf                       # Detailed analysis report
├── requirements.txt                 # Exact package dependency versions
└── README.md                        # Portfolio landing page & executive brief

## 1. Understanding the Data

The dataset contains *918 patients* with *11 clinical features* including age, cholesterol, resting blood pressure, chest pain type, and maximum heart rate.

### Class Distribution
The target variable (HeartDisease) is slightly imbalanced:
- *55.3%* of patients have heart disease
- *44.7%* do not

This mild imbalance is worth noting — it means accuracy alone isn't enough to evaluate model performance. ROC-AUC is a better metric here.

### Data Quality Issue
A significant data quality problem was found in the *Cholesterol* column: *172 patients had a value of 0, which is medically impossible. These were treated as missing values and replaced with the **median cholesterol value (237 mg/dL)*.

> The median was chosen over the mean to avoid the influence of extreme outliers on the imputed values.

---

## 2. Key Findings from Exploratory Data Analysis

### Most Important Features

Through visual analysis and correlation testing, four features stood out as the strongest predictors of heart disease:

| Feature | Why It Matters |
|---|---|
| *ST_Slope* | Abnormal ST slope during exercise is a well-known cardiac warning sign |
| *ExerciseAngina* | Chest pain triggered by exercise strongly indicates cardiovascular problems |
| *Oldpeak* | Higher ST depression during exercise signals abnormal heart behaviour |
| *MaxHR* | Patients with heart disease tend to have a lower maximum heart rate |

### Notable Patterns
- Patients *with* heart disease had significantly *higher Oldpeak* values
- Patients *with* heart disease had significantly *lower MaxHR* values
- *Chest pain type* showed a strong relationship with the target — not all chest pain types carry the same risk

---

## 3. Preprocessing Decisions

Before modelling, several preprocessing steps were applied:

- *Cholesterol zeros* replaced with median value
- *Categorical variables* converted using One-Hot Encoding (features expanded from 12 to 16)
- *Feature scaling* applied using StandardScaler — critical for SVM which is sensitive to feature magnitudes
- *Stratified 80/20 train-test split* to preserve the class ratio in both sets

---

## 4. Model Results

### SVM (Support Vector Machine)

An RBF kernel SVM was chosen to capture the non-linear relationships in the data.

| Metric | Score |
|---|---|
| Accuracy | 85.87% |
| ROC-AUC | *94.09%* |

The confusion matrix showed:
- 92 true positives (correctly identified heart disease)
- 66 true negatives (correctly identified no heart disease)
- 16 false positives
- 10 false negatives

> *False negatives are the most critical error in this context* — they represent patients with heart disease who were incorrectly classified as healthy. With only 10, the model performs well on this front.

### XGBoost

A baseline XGBoost model was trained first, then optimised using Grid Search Cross-Validation over:
- n_estimators: [50, 100, 150, 200, 250]
- learning_rate: [0.01, 0.05, 0.1, 0.2, 0.3]

*Best parameters found:* learning_rate=0.1, n_estimators=200

| Metric | Baseline | Tuned |
|---|---|---|
| Accuracy | 85.87% | 85.87% |
| ROC-AUC | 91.62% | 91.95% |

---

## 5. Model Comparison

| | SVM | XGBoost |
|---|---|---|
| Accuracy | 85.87% | 85.87% |
| ROC-AUC | *94.09%* | 91.95% |
| Interpretability | ❌ Black box | ✅ Feature importance |
| Speed | Slower on large data | Faster |
| Best for | Pure predictive performance | Explainable decisions |

### Which model is better?

*It depends on the use case:*

- If the goal is *maximum predictive accuracy* → SVM wins (higher ROC-AUC)
- If the goal is *clinical decision support* where doctors need to understand why a prediction was made → XGBoost wins (interpretable feature importance)

In a real healthcare setting, *XGBoost would likely be the preferred choice* because explainability builds trust with medical professionals and supports better patient outcomes.

---

## 6. Feature Importance (XGBoost)

The top features identified by XGBoost align closely with established medical knowledge:

1. *ST_Slope* — the most important predictor
2. *ExerciseAngina* — second strongest signal
3. *Oldpeak* — third strongest signal
4. *MaxHR* — fourth strongest signal

This alignment with clinical expertise gives confidence that the model is learning meaningful patterns rather than noise.

---

## 7. Conclusions

- Both models achieved *~86% accuracy* on unseen test data
- SVM showed stronger discriminative power (*ROC-AUC: 94.09%*)
- XGBoost provided more *interpretable results* through feature importance
- The most clinically significant features — ST_Slope, ExerciseAngina, Oldpeak — were consistently identified across all analyses
- Hyperparameter tuning improved XGBoost's cross-validation ROC-AUC from 91.62% to 91.95%

### Limitations
- Results are based on a single dataset of 918 patients
- Hyperparameter tuning was limited to two parameters (n_estimators, learning_rate)
- The model has not been validated on external datasets

### Future Work
- Test on larger and more diverse patient datasets
- Explore ensemble methods combining SVM and XGBoost
- Apply SHAP values for deeper model interpretability
- Consider additional hyperparameters (max_depth, subsample) in tuning
