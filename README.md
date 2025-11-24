# Loan Default Prediction

A machine learning project to predict whether a customer is likely to accept a personal loan offer, based on their demographics and financial data.

## Overview

This notebook walks through the end-to-end data science pipeline:
- Data exploration and preprocessing
- Feature analysis and selection
- Model training and tuning (Random Forest)
- Performance evaluation and feature importance

## Dataset

- **Source:** Bank Personal Loan Modeling dataset
- **Target Variable:** `Personal Loan` (0 = No, 1 = Yes)
- **File:** `Bank_Personal_Loan_Modelling.csv`

## Steps

### 1. Data Loading & Inspection
- Imported required libraries
- Loaded the dataset using `pandas`
- Checked shape, data types, and null values

### 2. Data Cleaning
- Replaced negative values in the `Experience` column with the median of valid values
- Dropped `ID` and `ZIP Code` as they are not useful for modeling

### 3. Exploratory Data Analysis
- Visualized distributions of numerical features using box and violin plots
- Correlation heatmap to observe relationships with target variable

### 4. Feature Engineering
- Applied One-Hot Encoding on `Family` and `Education` columns
- Used `StandardScaler` to normalize numerical features

### 5. Model Training
- Split data into training and test sets (80/20)
- Trained a `RandomForestClassifier` with hyperparameter tuning via `GridSearchCV`

### 6. Model Evaluation
- Evaluated model on test data using:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
- Plotted the confusion matrix

### 7. Feature Importance
- Ranked features based on importance from the trained Random Forest model
- Visualized the top 10 features

## Final Results

- **Model Used:** Random Forest Classifier
- **Best Parameters:** Grid search returned optimal hyperparameters
- **Top Features:** `Income`, `CD Account`, `Education`, `CCAvg`, `Mortgage`

## Libraries Used

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

