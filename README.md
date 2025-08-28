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
- `lgd_deploy_data.xlsx` — new data for prediction
- `README.md` — this file

---

## 🛠️ Deployment

The final model was deployed on new data (`lgd_deploy_data.xlsx`) using the same preprocessing steps:

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# 1. Load new data
deploy_data = pd.read_excel("lgd_deploy_data.xlsx")

# 2. Select final features
deploy_data = deploy_data[[
    'Income ($)', 'Loan Term (Months)', 'Debt to Income Ratio (%)',
    'Previous Defaults', 'Employment History (Years)', 'Exposure Amount ($)'
]]

# 3. Outlier capping
q1 = deploy_data.quantile(0.25)
q3 = deploy_data.quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

for col in deploy_data.columns:
    if deploy_data[col].dtypes != object:
        deploy_data[col] = np.where(deploy_data[col] > upper[col], upper[col], deploy_data[col])
        deploy_data[col] = np.where(deploy_data[col] < lower[col], lower[col], deploy_data[col])

# 4. Scale inputs
scaler = StandardScaler()
scaler.fit(deploy_data)
inputs_scaled = scaler.transform(deploy_data)
data_scaled = pd.DataFrame(inputs_scaled, columns=deploy_data.columns)

# 5. Predict LGD
reg = LinearRegression()
reg.fit(X_train_uni, y_train_uni)  # model was trained earlier
data_scaled['predicted_LGD'] = reg.predict(data_scaled)
