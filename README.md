# fraud-detection-ml

Analyzed 6.36M financial transactions to detect fraud using exploratory data analysis, statistical inference, and machine learning.

## Dataset

- **Source:** [Kaggle — Online Payments Fraud Detection](https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset)
- **Size:** 6,362,620 rows × 11 columns
- **Fraud rate:** ~0.13% (severe class imbalance)
- **Key features:** transaction type, amount, sender/receiver balances, `isFraud` target

## What We Did

**EDA & Data Cleaning**
- Checked for missing values, duplicates, and outliers (IQR + Z-score)
- Analyzed fraud distribution across transaction types and amounts
- Found fraud is exclusively in `TRANSFER` and `CASH_OUT` types
- Median fraud amount ≈ 441,423 vs non-fraud ≈ 74,685
- Engineered balance discrepancy features — fraudulent transactions consistently show abnormal origin/destination balance patterns

**Statistical Inference (α = 0.05)**
- Mann-Whitney U test — transaction amount vs fraud
- Pearson Chi-Square — transaction type vs fraud
- Welch's t-test — balance patterns vs fraud

**Feature Engineering**
- Log-transformed amounts, balance discrepancy flags, zero-balance indicators, one-hot encoded transaction types

**Modelling**
- Handled class imbalance via stratified split + SMOTE
- Trained Logistic Regression, Random Forest, and XGBoost
- Threshold tuning on XGBoost for precision/recall tradeoff

## Results

**Logistic Regression:** ROC-AUC 0.9826, Avg Precision 0.2207, and Accuracy 90.65%

**Random Forest:** ROC-AUC 0.9968, Avg Precision 0.7685, and Accuracy 94.29%

**XGBoost:** ROC-AUC 0.9993, Avg Precision 0.8565, and Accuracy 99.03%

Top predictors: `oldbalanceOrg`, `oldbalanceDest`, `amount`, `type_TRANSFER`, `type_CASH_OUT`

> Precision is low across all models due to extreme class imbalance — ROC-AUC and Average Precision are the meaningful metrics here.

## Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, SciPy, Imbalanced-learn (SMOTE)
