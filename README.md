This project aims to predict whether a telecom customer will churn (leave the company) using Machine Learning techniques. Customer churn prediction is an important business problem because retaining existing customers is more cost-effective than acquiring new ones.

Using the IBM Telco Customer Churn dataset, this project builds and compares multiple classification models to identify customers who are at high risk of churn.


Technologies Used -

1)Python
2)Pandas & NumPy
3)Matplotlib & Seaborn
4)Scikit-learn
5)XGBoost

The project follows a complete end-to-end machine learning pipeline:

1️⃣ Data Cleaning

Converted TotalCharges column to numeric format

Handled missing values using median imputation

Removed unnecessary column (customerID)

2️⃣ Feature Engineering

Encoded categorical variables using One-Hot Encoding

Converted target variable Churn into binary format (Yes = 1, No = 0)

3️⃣ Train-Test Split

Split dataset into 80% training and 20% testing

Used stratified sampling to maintain class distribution

4️⃣ Handling Class Imbalance

Applied SMOTE (Synthetic Minority Over-sampling Technique)

Balanced churn and non-churn classes in training data

5️⃣ Model Building

Logistic Regression (Baseline Model)

XGBoost Classifier (Advanced Gradient Boosting Model)

6️⃣ Model Evaluation

Models were evaluated using:

Accuracy Score

ROC-AUC Score

Confusion Matrix

Precision, Recall, and F1-Score

7️⃣ Hyperparameter Tuning

Used GridSearchCV to optimize:

max_depth

learning_rate

n_estimators

8️⃣ Model Saving

Saved trained XGBoost model using Joblib

Saved StandardScaler for deployment readiness
Imbalanced-learn (SMOTE)
