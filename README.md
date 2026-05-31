# Project Analysis — Heart Failure Prediction

## Overview

This project predicts whether a patient has heart disease using clinical data and compares the performance of two machine learning models: Support Vector Machine (SVM) and XGBoost.

**Dataset:** [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data) — 918 patients, 11 clinical features.

---

## Project Structure

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
```

---

## 1. Understanding the Data

The dataset contains **918 patients** with **11 clinical features** including age, cholesterol, resting blood pressure, chest pain type, and maximum heart rate.

### Class Distribution

The target variable (**HeartDisease**) is slightly imbalanced:

- **55.3%** of patients have heart disease
- **44.7%** do not

This mild imbalance means accuracy alone is not enough to evaluate model performance. ROC-AUC provides a more reliable assessment.

### Data Quality Issue

A significant data quality problem was found in the **Cholesterol** column:

**172 patients had a value of 0, which is medically impossible. These values were treated as missing and replaced using the median cholesterol value (237 mg/dL).**

> The median was chosen over the mean to reduce the influence of extreme outliers.

---

## 2. Key Findings from Exploratory Data Analysis

### Most Important Features

| Feature | Why It Matters |
|----------|----------|
| **ST_Slope** | Abnormal ST slope during exercise is a well-known cardiac warning sign |
| **ExerciseAngina** | Chest pain triggered by exercise strongly indicates cardiovascular problems |
| **Oldpeak** | Higher ST depression during exercise signals abnormal heart behaviour |
| **MaxHR** | Patients with heart disease tend to have a lower maximum heart rate |

### Notable Patterns

- Patients with heart disease had significantly higher **Oldpeak** values.
- Patients with heart disease had significantly lower **MaxHR** values.
- **Chest pain type** showed a strong relationship with the target variable.

---

## 3. Preprocessing Decisions

Before modelling, several preprocessing steps were applied:

- Replaced cholesterol values of 0 with the median value.
- Converted categorical variables using One-Hot Encoding.
- Expanded features from 12 to 16 dimensions.
- Applied StandardScaler feature scaling.
- Used a stratified 80/20 train-test split.

---

## 4. Model Results

### Support Vector Machine (SVM)

An RBF kernel SVM was used to capture non-linear relationships in the data.

| Metric | Score |
|----------|----------|
| Accuracy | 85.87% |
| ROC-AUC | **94.09%** |

#### Confusion Matrix

- 92 True Positives
- 66 True Negatives
- 16 False Positives
- 10 False Negatives

> False negatives are the most critical error because they represent patients with heart disease incorrectly classified as healthy.

### XGBoost

A baseline XGBoost model was trained and then optimized using Grid Search Cross Validation.

#### Hyperparameter Search

```python
n_estimators = [50, 100, 150, 200, 250]
learning_rate = [0.01, 0.05, 0.1, 0.2, 0.3]
```

**Best Parameters**

```python
learning_rate = 0.1
n_estimators = 200
```

| Metric | Baseline | Tuned |
|----------|----------|----------|
| Accuracy | 85.87% | 85.87% |
| ROC-AUC | 91.62% | 91.95% |

---

## 5. Model Comparison

| Metric | SVM | XGBoost |
|----------|----------|----------|
| Accuracy | 85.87% | 85.87% |
| ROC-AUC | **94.09%** | 91.95% |
| Interpretability | ❌ Low | ✅ High |
| Speed | Slower | Faster |
| Best Use Case | Predictive Performance | Explainable Predictions |

### Which Model Is Better?

**SVM** achieved the highest ROC-AUC score and delivered the strongest predictive performance.

**XGBoost** provided feature importance information, making it easier to explain predictions and support real-world clinical decision-making.

For healthcare applications where explainability is important, XGBoost would likely be preferred despite the slightly lower ROC-AUC.

---

## 6. Feature Importance (XGBoost)

The most influential features were:

1. **ST_Slope**
2. **ExerciseAngina**
3. **Oldpeak**
4. **MaxHR**

These findings align with established clinical knowledge, increasing confidence that the model is learning meaningful patterns.

---

## 7. Conclusions

### Key Findings

- Both models achieved approximately **86% accuracy**.
- SVM produced the strongest ROC-AUC (**94.09%**).
- XGBoost offered superior interpretability.
- ST_Slope, ExerciseAngina, Oldpeak, and MaxHR consistently emerged as the most predictive features.
- Hyperparameter tuning improved XGBoost ROC-AUC from **91.62%** to **91.95%**.

### Limitations

- Based on a single dataset of 918 patients.
- Hyperparameter tuning only explored two parameters.
- No external dataset validation was performed.

### Future Work

- Test on larger and more diverse datasets.
- Explore ensemble methods combining SVM and XGBoost.
- Apply SHAP values for explainable AI.
- Tune additional XGBoost parameters such as:
  - `max_depth`
  - `subsample`
  - `colsample_bytree`

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Matplotlib
- Seaborn

---

## Results Summary

| Model | Accuracy | ROC-AUC |
|----------|----------|----------|
| SVM | 85.87% | **94.09%** |
| XGBoost (Tuned) | 85.87% | 91.95% |

**Best Overall Predictive Model:** SVM  
**Best Explainable Model:** XGBoost
