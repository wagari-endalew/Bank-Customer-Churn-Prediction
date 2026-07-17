# 🏦 Bank Customer Churn Prediction

Predicting which bank customers are likely to churn, using supervised machine learning to support proactive retention strategy.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Problem Statement

Customer churn — when a client closes their account or stops using the bank's services — is one of the costliest problems in retail banking, since acquiring a new customer is far more expensive than retaining an existing one. This project builds a classification model that predicts whether a customer will **exit (churn)**, based on their demographic profile and account activity, so the bank can proactively target at-risk customers with retention offers.

---

## 🗂 Dataset

**10,000 customer records** across 14 columns, with no missing values in any column.

| Feature | Description |
|---|---|
| `CustomerId` | Unique customer identifier |
| `CreditScore` | Customer's credit score |
| `Geography` | Customer's country (France, Germany, Spain) |
| `Gender` | Customer's gender |
| `Age` | Customer's age |
| `Tenure` | Years as a bank customer |
| `Balance` | Account balance |
| `NumOfProducts` | Number of bank products used |
| `HasCrCard` | Whether the customer has a credit card (1/0) |
| `IsActiveMember` | Whether the customer is an active member (1/0) |
| `EstimatedSalary` | Customer's estimated salary |
| `Exited` | **Target variable** — 1 if the customer churned, 0 otherwise |

**Class balance:** 79.6% retained (7,963) vs. **20.4% churned (2,037)** — a moderate class imbalance that informed the choice of ROC-AUC (rather than raw accuracy) as the primary model-selection metric.

> This dataset is used for educational/portfolio purposes. If adapting this pipeline for production use with real customer data, ensure compliance with data privacy regulations (e.g. GDPR).

---

## 🔬 Methodology

1. **Data Cleaning & Preprocessing** — verified no missing values; dropped non-predictive identifiers (`RowNumber`, `CustomerId`, `Surname`) from the modeling features.
2. **Exploratory Data Analysis** — distribution plots (KDE) of numeric features by churn status, pairplots, violin plots (Age), count plots (Tenure, Geography, Gender, HasCrCard, IsActiveMember), churn-ratio donut charts by gender, and a correlation heatmap.
3. **Feature Engineering** — created five new features to give the models more signal:
   - `Balance_per_product` — account balance divided by number of products held
   - `salary_to_balance_ratio` — estimated salary relative to account balance
   - `age_group` — binned age into 6 buckets (`<25` → `65+`)
   - `tenure_bucket` — binned tenure into 5 buckets (`0` → `10+`)
   - `high_balance` — binary flag for customers in the top 25% of account balance
4. **Preprocessing Pipeline** — built with `ColumnTransformer`: numeric features imputed (median) and scaled (`StandardScaler`); categorical features imputed (most frequent) and one-hot encoded, all wrapped in a single reusable `sklearn.Pipeline`.
5. **Model Building** — trained and compared 5 classifiers via 5-fold `StratifiedKFold` cross-validation, scored on ROC-AUC:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
   - AdaBoost
   - Support Vector Classifier (SVC)
6. **Model Selection & Evaluation** — selected the model with the highest mean CV ROC-AUC, refit it on the full training set, and evaluated on a held-out 20% test set (stratified split, `random_state=42`).
7. **Model Export** — serialized the winning pipeline with `joblib` → `best_churn_pipeline.pkl`.

---

## 📊 Results

**Best model:** `Gradient Boosting` — selected via 5-fold cross-validated ROC-AUC (mean CV ROC-AUC: 0.8622), confirmed on the held-out test set (ROC-AUC: 0.8687).

**Held-out test set performance:**

| Metric | Score |
|---|---|
| Accuracy | **86.45%** |
| Precision (churn class) | **76.98%** |
| Recall (churn class) | **47.67%** |
| F1-Score (churn class) | **58.88%** |
| ROC-AUC | **86.87%** |

```
              precision    recall  f1-score   support

           0       0.88      0.96      0.92      1593
           1       0.77      0.48      0.59       407

    accuracy                           0.86      2000
   macro avg       0.82      0.72      0.75      2000
weighted avg       0.86      0.86      0.85      2000
```

The model is strong at identifying customers who will **stay** (96% recall on class 0) but more conservative at flagging churners (48% recall on class 1) — it catches churners with high precision (77%) when it does flag them, but misses roughly half of actual churn cases. For a retention-targeting use case, this is a reasonable trade-off: false positives (offering a retention incentive to a loyal customer) are cheap, while false negatives (missing a real churner) are the costlier error — so **improving recall on the churn class is the top priority for future iterations** (see below).

### 🔑 Top Predictive Features

Feature importance from the best model highlighted these as the strongest churn signals:

| Rank | Feature | Importance |
|---|---|---|
| 1 | `Age` | 32.8% |
| 2 | `NumOfProducts` | 26.7% |
| 3 | `Balance_per_product` | 6.7% |
| 4 | `IsActiveMember` | 5.7% |
| 5 | `Balance` | 5.1% |
| 6 | `Geography_Germany` | 5.1% |

**Key insights:**
- **Age is the single strongest churn predictor** — older customers churn at a materially higher rate than younger ones.
- **Number of products matters more than balance itself** — customers holding only 1 product are meaningfully more likely to churn than those with 2+, suggesting cross-sell/product engagement is a retention lever.
- **Inactive members churn more** — `IsActiveMember` ranks in the top 5 features, confirming that engagement (not just tenure) predicts loyalty.
- **Germany stands out geographically** — `Geography_Germany` is a top-6 feature, indicating elevated churn risk in that market relative to France and Spain.

---

## 🧰 Tech Stack

- **Language:** Python 3.11
- **Data handling:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Modeling:** scikit-learn
- **Model persistence:** joblib
- **Environment:** Jupyter Notebook (VS Code / Google Colab compatible)

---

## 📁 Project Structure

```
Bank-Customer-Churn-Prediction/
│
├── Churn_Custemer_ML_Last_Prediction.ipynb   # Main analysis & modeling notebook
├── Churn_Modelling.csv                        # Raw dataset
├── best_churn_pipeline.pkl                    # Serialized best-performing model pipeline
├── requirements.txt                           # Project dependencies
├── LICENSE
└── README.md
```

---

## 🚀 How to Run

### Option A — Google Colab
1. Open the notebook directly in [Google Colab](https://colab.research.google.com/).
2. Upload `Churn_Modelling.csv` when prompted, or mount Google Drive.
3. Run all cells sequentially.

### Option B — Local (VS Code / Jupyter)
```bash
# Clone the repo
git clone https://github.com/wagari-endalew/Bank-Customer-Churn-Prediction.git
cd Bank-Customer-Churn-Prediction

# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1      # Windows PowerShell
# source venv/bin/activate       # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook Churn_Custemer_ML_Last_Prediction.ipynb
```

### Using the saved model
```python
import joblib
import pandas as pd

model = joblib.load("best_churn_pipeline.pkl")
prediction = model.predict(new_customer_df)          # 0 = stays, 1 = churns
probability = model.predict_proba(new_customer_df)[:, 1]  # churn probability
```

Note: any new input data must go through the **same feature engineering steps** used in training (`Balance_per_product`, `salary_to_balance_ratio`, `age_group`, `tenure_bucket`, `high_balance`) before being passed to the pipeline.

---

## 🔮 Future Improvements

- **Improve churn-class recall** — currently 48%; address the class imbalance more directly with `class_weight='balanced'`, SMOTE oversampling, or threshold tuning on `predict_proba` to catch more true churners at the cost of some precision.
- Add SHAP-based model explainability for individual customer-level predictions.
- Deploy as a REST API (FastAPI/Flask) for real-time scoring.
- Build a lightweight dashboard (Streamlit) so business users can explore at-risk customer segments directly.

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## 👤 Author

**Wagari Endalew**
[GitHub](https://github.com/wagari-endalew)
