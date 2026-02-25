**GenAI Capstone Project**
Credit Risk Prediction using Machine Learning

**Abstract**
This project builds a Machine Learning model to predict loan default risk using borrower financial
and demographic data. The model classifies applicants as low-risk or high-risk and estimates
probability of default using Logistic Regression.
Problem Statement
Financial institutions need accurate systems to reduce losses caused by loan defaults. This project
aims to predict whether a borrower will default based on structured financial data.

**Dataset Features**
• person_age
• person_income
• person_home_ownership
• person_emp_length
• loan_amnt
• loan_int_rate
• loan_percent_income
• cb_person_default_on_file
• cb_person_cred_hist_length
• Target: loan_status (0 = No Default, 1 = Default)

**Project Workflow**
• Data Cleaning and missing value handling
• Exploratory Data Analysis
• Feature Encoding and Selection
• Feature Scaling using StandardScaler
• Train-Test Split
• Model Training using Logistic Regression and Decision Tree
• Evaluation using Accuracy, Confusion Matrix and ROC-AUC

**Business Interpretation**
Risk levels can be defined using probability thresholds: Low Risk (<0.2), Medium Risk (0.2–0.5),
High Risk (>0.5). This makes the model suitable for real-world lending decisions.
How to Run the Project
git clone https://github.com/your-username/GenAI_Capstone.git
Open GenAI_Capstone.ipynb in Google Colab
Run all cells sequentially

**Future Improvements**
• Hyperparameter tuning
• Cross-validation
• ROC curve visualization
• Feature importance analysis
• Model deployment using Streamlit

**Author: Palak, Samarth Sangtani , Jashvitha Omkaram Laxmi
GenAI Capstone Project | Machine Learning**
