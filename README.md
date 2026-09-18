# 🛡️ AI Insurance Risk Prediction

### Intelligent Insurance Fraud Detection & Risk Analysis Platform

> A Machine Learning and Explainable AI project for detecting potential insurance fraud, estimating fraud probability, prioritizing high-risk claims, and explaining model predictions.

---

## 📌 Overview

**AI Insurance Risk Prediction** is a Machine Learning project designed to analyze insurance claims and identify potentially fraudulent cases.

The project combines **Data Science, Machine Learning, Explainable AI and Streamlit** to build an end-to-end insurance fraud detection solution.

The system can:

- 📊 Analyze insurance claim data
- 🧹 Clean and preprocess raw data
- ⚙️ Perform feature engineering
- 🤖 Train and compare Machine Learning models
- 🎯 Optimize the classification threshold
- 🚨 Detect potentially fraudulent claims
- 📈 Estimate fraud probabilities
- 🟢🟠🔴 Assign risk levels
- 🔍 Explain predictions using SHAP
- 📋 Prioritize claims requiring further investigation
- 📥 Export prediction results as CSV
- 🎨 Provide an interactive Streamlit dashboard

---

# 🎯 Project Objectives

The main objectives of this project are to:

1. Understand and explore insurance claim data.
2. Identify patterns associated with fraudulent claims.
3. Build a Machine Learning model for fraud detection.
4. Compare several classification algorithms.
5. Optimize the decision threshold for fraud detection.
6. Generate a fraud probability for each claim.
7. Assign an interpretable risk level.
8. Explain individual predictions using SHAP.
9. Provide an interactive interface for insurance risk analysis.

---

# 🧠 Machine Learning Approach

Several Machine Learning algorithms were evaluated:

- Logistic Regression
- Random Forest
- Optimized Random Forest
- XGBoost

The final solution uses an **optimized Random Forest classifier**.

### Model configuration

```text
Algorithm        : Random Forest
Task             : Binary Classification
Target           : fraud_reported
Class 0          : Non-fraud
Class 1          : Fraud
Class weighting  : Balanced
Decision threshold: 0.20
```

The Random Forest model was optimized using RandomizedSearchCV with cross-validation.

**Best hyperparameters**
```text
n_estimators      = 700
max_depth         = 5
min_samples_split = 2
min_samples_leaf  = 8
max_features      = None
```

---
# 📊 Dataset

The main dataset used for fraud detection is the **Insurance Claims** dataset.

### Main dataset
```text
Rows       : 1,000
Variables  : 40 initially
Target     : fraud_reported
```
Target distribution:
| Class           | Count | Percentage |
| --------------- | ----: | ---------: |
| Non-Fraud (`N`) |   753 |      75.3% |
| Fraud (`Y`)     |   247 |      24.7% |

The project also contains additional insurance datasets used for exploratory analysis and experimentation:

- train_auto.csv
- test_auto.csv
- fraud_oracle.csv

---
# ⚙️ Data Preparation

The preprocessing pipeline includes:

#### Missing values

Missing values represented by *?* are converted to *NaN*.

Categorical missing values are handled using imputation.

#### Numerical features
```text
Median Imputation
        ↓
StandardScaler
```

#### Categorical features
```tex
Most Frequent Imputation
        ↓
OneHotEncoder
```

Unknown categories are handled using:
```text
handle_unknown="ignore"
```

---
# 🔧 Feature Engineering

Several additional features were created from the original variables.

#### Date features
- policy_bind_year
- policy_bind_month
- incident_year
- incident_month

#### Policy duration
```text
policy_duration_days
```

Represents the number of days between policy binding and the incident.

#### Vehicle age
```text
vehicle_age
```

Calculated from the vehicle year and incident year.

#### Incident period

The incident hour is transformed into four categories:
```text
Morning
Afternoon
Evening
Night
```
