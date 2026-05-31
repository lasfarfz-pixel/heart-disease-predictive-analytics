# Heart Failure Prediction — ML Classification Pipeline

> Exploratory Data Analysis · SVM · XGBoost · Hyperparameter Tuning · Healthcare Analytics

-----

## Project Overview

A complete end-to-end machine learning pipeline to predict heart disease from clinical patient data. This project covers the full data science workflow: raw data exploration, preprocessing, model training, evaluation, and hyperparameter optimization — comparing two powerful classification algorithms (SVM and XGBoost).

The goal is to build models that can assist in *early diagnosis and risk prediction* of cardiovascular disease, one of the most prevalent health issues worldwide.

*Dataset:* [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data) — 918 patients, 11 clinical features.

-----

## Project Structure


heart-failure-prediction/
│
├── heart-failure-prediction.ipynb   # Full pipeline notebook
├── report.pdf                       # Detailed analysis report
└── README.md                        # This file


-----

## Key Findings

### EDA Insights

|Observation         |Detail                                                               |
|--------------------|---------------------------------------------------------------------|
|Dataset balance     |55.3% positive (Heart Disease = 1) — mild imbalance                  |
|Data quality issue  |172 zero values in Cholesterol → replaced with median (237)          |
|Strongest predictors|ST_Slope, ExerciseAngina, Oldpeak, MaxHR                             |
|Notable pattern     |Higher Oldpeak and lower MaxHR strongly associated with heart disease|

### Model Performance

|Model             |Accuracy|ROC-AUC                        |
|------------------|--------|-------------------------------|
|SVM (RBF kernel)  |85.87%  |94.09%                         |
|XGBoost (baseline)|85.87%  |91.62%                         |
|XGBoost (tuned)   |85.87%  |91.95% (GridSearchCV optimised)|


> SVM achieves higher ROC-AUC, indicating stronger discrimination ability. XGBoost offers better interpretability through feature importance analysis — a critical advantage in healthcare applications.

-----

## Methodology

### 1 — EDA & Preprocessing

- Identified and handled data quality issues (zero-value imputation)
- Performed univariate and bivariate analysis using histograms, boxplots, and heatmaps
- Applied One-Hot Encoding (12 → 16 features) and StandardScaler
- Stratified 80/20 train-test split to preserve class distribution

### 2 — SVM Classification

- Trained an RBF kernel SVM — chosen for its ability to model non-linear relationships
- Evaluated using accuracy, ROC-AUC, confusion matrix, and classification report
- Created a 3D PCA visualisation of the decision boundary

### 3 — XGBoost + Hyperparameter Tuning

- Trained a baseline XGBoost classifier
- Applied GridSearchCV over n_estimators [50–250] and learning_rate [0.01–0.3]
- Best parameters: learning_rate=0.1, n_estimators=200 (5-fold CV ROC-AUC: 0.9195)
- Analysed feature importance to interpret model predictions

-----

## Most Important Features

Based on XGBoost feature importance analysis:

1. *ST_Slope* — abnormal slopes strongly linked to heart disease
1. *ExerciseAngina* — exercise-induced chest pain is a key diagnostic indicator
1. *Oldpeak* — elevated ST depression during exercise signals abnormal heart behaviour
1. *MaxHR* — lower maximum heart rate reflects reduced cardiovascular function

-----

## Tech Stack

|Library                 |Purpose                                      |
|------------------------|---------------------------------------------|
|pandas / numpy      |Data manipulation                            |
|matplotlib / seaborn|Visualisation                                |
|scikit-learn          |SVM, preprocessing, GridSearchCV, evaluation |
|xgboost               |Gradient boosting classifier                 |
|PCA                   |Dimensionality reduction for 3D visualisation|

-----

## How to Run

1. Clone the repository
   
   bash
   git clone https://github.com/lasfarfz-pixel/heart-failure-prediction.git
   
1. Install dependencies
   
   bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
   
1. Download heart.csv from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data) and place it in the project folder
1. Open the notebook and run all cells in order

> *Note:* Replace the Google Colab file upload cell with df = pd.read_csv('heart.csv') when running locally.

-----

