# Credit LGD Predictor 🧠📉

This project builds a full machine learning pipeline to predict **Loss Given Default (LGD%)** for consumer loans. It includes data cleaning, feature engineering, statistical analysis, model training, and deployment on new data.

---

## 🚀 Project Summary

- **Goal**: Predict LGD% for loan applicants using structured credit data
- **Model**: Linear Regression
- **Target Variable**: LGD%
- **Final Features Used**:
  - Income ($)
  - Loan Term (Months)
  - Debt to Income Ratio (%)
  - Previous Defaults
  - Employment History (Years)
  - Exposure Amount ($)

---

## 🧪 Workflow Overview

### 1. Data Preprocessing
- Outlier treatment via IQR capping
- Missing value check (none found)
- Feature engineering:
  - `Income_mean_by_Region`
  - `Previous_Defaults_mean_by_Credit_Score`
- Categorical encoding with `pd.get_dummies(drop_first=True)`
- Scaling with `StandardScaler`

### 2. Statistical Analysis
- Normality check via Kolmogorov-Smirnov test
- Correlation analysis (target threshold: 10%)
- Intercorrelation filtering (threshold: 60%)
- VIF filtering (kept features with VIF < 9)
- Univariate R² analysis to select top predictors

### 3. Model Training
- Train-test split
- Linear Regression fit on scaled inputs
- Evaluation:
  - **Train R²**: ~0.49
  - **Test R²**: ~0.48
  - MAE, MSE, RMSE also computed
  - Adjusted R² calculated for both sets

---

## 📦 Files Included

- `credit-lgd-predictor.pdf` — full project walkthrough
- `model.pkl` — trained regression model
- `scaler.pkl` — fitted StandardScaler
- `feature_names.pkl` — expected input columns
- `lgd_deploy_data.xlsx` — new data for prediction
- `README.md` — this file

---

## 🛠️ Deployment Instructions

```python
import pandas as pd
import pickle

# Load model and scaler
with open("model.pkl", "rb") as f:
    reg = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_names.pkl", "rb") as f:
    expected_columns = pickle.load(f)

# Load new data
deploy_data = pd.read_excel("lgd_deploy_data.xlsx")

# Preprocess
deploy_dummies = pd.get_dummies(deploy_data, drop_first=True)
deploy_dummies = deploy_dummies.reindex(columns=expected_columns, fill_value=0)
X_scaled = scaler.transform(deploy_dummies)

# Predict LGD%
y_pred = reg.predict(X_scaled)
deploy_data['lgd_prediction'] = y_pred
