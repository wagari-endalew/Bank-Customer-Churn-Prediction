# Bank Customer Churn Prediction

## Project Overview
This project predicts customer churn for a bank using Machine Learning. The goal is to identify customers who are likely to leave, helping the bank improve retention strategies.

The project is implemented in **Python** using **Google Colab**, covering data preprocessing, exploratory data analysis (EDA), feature engineering, model building, and evaluation.



## Dataset
The dataset contains customer information including demographic and account details. Key features include:

- CustomerID
- CreditScore
- Geography
- Gender
- Age
- Tenure
- Balance
- NumOfProducts
- HasCrCard
- IsActiveMember
- EstimatedSalary
- Exited (target variable)

> Note: Use anonymized or synthetic datasets to avoid sensitive data issues.


## Methodology
1. Data Preprocessing: Handle missing values, encode categorical variables, scale features.
2. Exploratory Data Analysis (EDA): Visualize distributions, correlations, and trends.
3. Model Building: Train models such as Logistic Regression, Random Forest, and XGBoost.
4. Model Evaluation: Use metrics like Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
5. Prediction & Insights: Identify high-risk churn customers and suggest retention strategies.


## How to Run
1. Open the notebook in **Google Colab**.
2. Run all cells sequentially.
3. Install required packages if needed:

```bash
!pip install pandas numpy scikit-learn matplotlib seaborn xgboost
