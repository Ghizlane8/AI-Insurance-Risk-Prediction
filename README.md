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

---
# 🚫 Data Leakage Prevention

For early fraud detection, claim amount variables are excluded from the predictive features:
```text
total_claim_amount
injury_claim
property_claim
vehicle_claim
```

Other non-predictive or identifier-related fields are also excluded:
```text
policy_number
incident_location
insured_zip
policy_bind_date
incident_date
_c39
```

This allows the model to focus on information available for the fraud-risk analysis rather than directly relying on claim amounts.

---
# 📈 Model Evaluation

The models were evaluated using several metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

#### Model comparison

| Model                   | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
| ----------------------- | --------: | -----: | -------: | ------: | -----: |
| Logistic Regression     |      0.60 |   0.67 |     0.63 |  0.8309 | 0.5260 |
| Random Forest           |      0.66 |   0.78 |     0.71 |  0.8268 | 0.5260 |
| Optimized Random Forest |      0.64 |   0.86 |     0.73 |  0.8366 | 0.6265 |
| XGBoost                 |      0.59 |   0.49 |     0.53 |  0.8357 | 0.5688 |

For fraud detection, particular attention was given to **Recall, F1-score and PR-AUC** because of the class imbalance.

---
# 🎚️ Decision Threshold

The default classification threshold of *0.50* was adjusted during the model evaluation process.

The project uses a saved threshold of:
```text
0.20
```

The prediction is calculated as:
```text
prediction = (probability >= threshold).astype(int)
```

This threshold is used to determine the predicted class from the estimated fraud probability.

---
# 🚨 Risk Scoring

Each claim receives a fraud probability between *0* and *1*.

The application converts this probability into an interpretable risk level:
| Fraud Probability | Risk Level  |
| ----------------: | ----------- |
|           `< 20%` | 🟢 Low      |
|     `20% – < 50%` | 🟠 Moderate |
|           `≥ 50%` | 🔴 High     |

The system also generates a final prediction:
```text
0 → Non-Fraud
1 → Fraud
```

---
# 🔍 Explainable AI

The project integrates **SHAP (SHapley Additive exPlanations)** to make individual predictions more interpretable.

The SHAP module uses:
```text
shap.TreeExplainer()
```

to analyze the contribution of each transformed feature to the prediction.

For a selected claim, the dashboard displays:

- 🔴 Factors increasing the risk
- 🟢 Factors reducing the risk
- 📊 SHAP contribution of the main variables
- 🔎 Main risk factor
- 🎯 Fraud probability
- ⚠️ Risk level
- 📌 Prediction decision

#### Interpretation
```text
SHAP > 0
→ contributes toward the fraud prediction

SHAP < 0
→ contributes away from the fraud prediction
```

---
# 🎨 Streamlit Dashboard

The project includes an interactive dashboard developed with **Streamlit**.

The application allows users to upload a CSV file and analyze insurance claims.

### Dashboard workflow
```text
                CSV File
                   │
                   ▼
          Data Preparation
                   │
                   ▼
          Machine Learning
                   │
                   ▼
        Fraud Probability
                   │
                   ▼
            Risk Scoring
                   │
          ┌────────┴────────┐
          ▼                 ▼
     Prioritization       SHAP / XAI
          │                 │
          └────────┬────────┘
                   ▼
          Interactive Dashboard
```

## 📊 Dashboard Features
### 📁 CSV Upload

Users can import an insurance claims CSV file directly into the application.

## 📈 Overview KPIs

The dashboard displays:

- Number of analyzed claims
- Number of high-risk claims
- Predicted fraud rate
- Average fraud probability

## 🎯 Risk Distribution

Claims are distributed into:
```text
🟢 Low Risk
🟠 Moderate Risk
🔴 High Risk
```

## 📊 Fraud Probability Distribution

The dashboard provides a distribution of fraud probabilities:
```text
0–20%
20–40%
40–60%
60–80%
80–100%
```

## 🚨 Claims to Review

Users can select:
```text
All claims
Top 5
Top 10
Top 15
```
The claims are ordered according to their estimated fraud probability.

## 📊 Risk Score Profile

A boxplot provides a statistical view of the probability distribution using:

- Q1
- Median
- Q3
- Overall score dispersion

## 🔎 Individual Claim Explanation

The dashboard allows the user to select a claim from the prioritized cases.

The system then provides an individual explanation using SHAP.

Example workflow:
```text
Select Claim
     ↓
Fraud Probability
     ↓
Risk Level
     ↓
Prediction
     ↓
Top Risk Factors
     ↓
Protective Factors
     ↓
SHAP Contributions
```
This makes the model prediction easier to interpret and analyze.

## 📥 Export Results

Prediction results can be exported as a CSV file.

The exported results contain information such as:
```text
Fraud Probability (%)
Prediction
Risk Level
Decision
```
The file can then be used for further analysis.

# 📓 Notebooks
#### 01_fraud_detection_model.ipynb

Main Machine Learning notebook containing:

- Data loading
- Dataset understanding
- Exploratory Data Analysis
- Missing value analysis
- Target analysis
- Feature Engineering
- Data preprocessing
- Train/Test split
- Logistic Regression
- Random Forest
- Random Forest optimization
- XGBoost
- Model comparison
- Threshold optimization
- SHAP analysis
- Model saving

#### 02_fraud_prediction_test.ipynb

Notebook dedicated to testing the saved model.

It includes:

- Loading the trained model
- Loading the decision threshold
- Preparing input data
- Testing individual predictions
- Testing multiple claims
- Probability analysis

#### 03_prediction_module_test.ipynb

Notebook dedicated to testing the reusable prediction module:
```text
from src.prediction import predict_fraud
```
It verifies predictions on multiple insurance claims.

---
# 🛠️ Technologies
| Category            | Technologies                |
| ------------------- | --------------------------- |
| Programming         | Python                      |
| Data Processing     | Pandas, NumPy               |
| Machine Learning    | Scikit-learn, XGBoost       |
| Model               | Random Forest               |
| Explainable AI      | SHAP                        |
| Visualization       | Matplotlib, Seaborn, Plotly |
| Web Application     | Streamlit                   |
| Model Serialization | Joblib                      |
| Environment         | Python Virtual Environment  |

---
# 🚀 Installation
1. Clone the repository
```text
git clone https://github.com/YOUR_USERNAME/AI-Insurance-Risk-Prediction.git
```
```text
cd AI-Insurance-Risk-Prediction
```
2. Create a virtual environment

Windows
```text
python -m venv .venv
```

Activate it:
```text
.venv\Scripts\activate
```
Linux / macOS
```text
python3 -m venv .venv
```
```text
source .venv/bin/activate
```

3. Install dependencies
```text
pip install -r requirements.txt
```

# ▶️ Run the Application

From the project root:
```text
streamlit run app/app.py
```        
The Streamlit application will then open in your browser.

## 📋 Example Workflow
```text
1. Launch the Streamlit application
              ↓
2. Upload insurance_claims.csv
              ↓
3. Click "Lancer l'analyse de risque"
              ↓
4. View KPIs and risk distribution
              ↓
5. Explore high-risk claims
              ↓
6. Select a claim
              ↓
7. Analyze SHAP explanations
              ↓
8. Export prediction results
```

## 🔮 Future Improvements

Potential improvements include:

 - Probability calibration
 - External validation on new insurance datasets
 - Model monitoring
 - Data drift detection
 - Model performance monitoring
 - API deployment
 - Database integration
 - Authentication
 - Claim history management
 - Automated PDF reports
 - Cloud deployment
 - Advanced model monitoring

---
# 👩‍💻 Author

**Baali Ghizlane**

Data Science & Artificial Intelligence

**Areas of interest**
- Data Science
- Machine Learning
- Artificial Intelligence
- Explainable AI
- Data Analytics
- Insurance Analytics

# ⭐ Project

**AI Insurance Risk Prediction**

Machine Learning project focused on insurance fraud detection, risk scoring and Explainable AI.
