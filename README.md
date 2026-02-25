# 📋 Credit Risk Evaluator

A machine learning-powered web application that predicts loan default risk using a Decision Tree Classifier trained on 32,581 historical loan records.

🔗 **Live App:** [https://genaicapstone-a7eipdbqudn2niewt9s2mp.streamlit.app/](https://genaicapstone-a7eipdbqudn2niewt9s2mp.streamlit.app/)

---

## Overview

This project builds a **binary classification model** to predict whether a borrower will default on a loan. It takes applicant information (age, income, employment history) and loan details (amount, grade, interest rate, purpose) as inputs and outputs a risk assessment with default probability.

The model is deployed as an interactive **Streamlit web application** with real-time predictions.

---

## Features

- 🎯 **Real-time risk prediction** — enter applicant details and get an instant credit risk assessment
- 📊 **Three-tier decision output** — Approved (< 20% risk), Needs Review (20–50%), Rejected (> 50%)
- 🌗 **Light & dark mode** — responsive design that adapts to system theme
- ⚡ **Cached model training** — model trains once and is cached for fast subsequent predictions

---

## Model Performance

| Metric                    | Value      |
| ------------------------- | ---------- |
| Accuracy                  | **90.81%** |
| ROC-AUC                   | 0.6817     |
| F1-Score (Default Class)  | 0.79       |
| Precision (Default Class) | 0.81       |
| Recall (Default Class)    | 0.77       |

Two models were compared — Logistic Regression (84.5% accuracy, 0.86 ROC-AUC) and Decision Tree (90.8% accuracy, 0.68 ROC-AUC). The **Decision Tree** was selected for deployment due to higher accuracy and better recall on the default class.

> See [REPORT.md](REPORT.md) for the full technical report with detailed evaluation, EDA insights, and methodology.

---

## Dataset

- **Records:** 32,581 loan applications
- **Features:** 11 input features (numerical + categorical)
- **Target:** `loan_status` (0 = Non-Default, 1 = Default)
- **Class Split:** ~78% Non-Default, ~22% Default

Key features include: `person_age`, `person_income`, `person_emp_length`, `person_home_ownership`, `loan_intent`, `loan_grade`, `loan_amnt`, `loan_int_rate`, `loan_percent_income`, `cb_person_default_on_file`, `cb_person_cred_hist_length`.

---

## Tech Stack

| Component     | Technology                                   |
| ------------- | -------------------------------------------- |
| Language      | Python 3.x                                   |
| ML            | scikit-learn (Decision Tree, StandardScaler) |
| Data          | pandas, NumPy                                |
| Visualization | Matplotlib, Seaborn                          |
| Web App       | Streamlit                                    |
| Deployment    | Streamlit Cloud                              |

---

## Project Structure

```
GenAI_Capstone/
├── app.py                    # Streamlit web application
├── GenAI_Capstone.ipynb      # Jupyter notebook (EDA + model training)
├── credit_risk_dataset.csv   # Training dataset
├── requirements.txt          # Python dependencies
├── REPORT.md                 # Full technical report
└── README.md                 # This file
```

---

## Run Locally

```bash
# Clone the repository
git clone https://github.com/PALAK7890/GenAI_Capstone.git
cd GenAI_Capstone

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## Team

| Member      | Contribution                                                           |
| ----------- | ---------------------------------------------------------------------- |
| **Palak**   | Dataset sourcing, data preprocessing, EDA, model training & evaluation |
| **Samarth** | Streamlit app development, UI/UX design, deployment, bug fixes         |

---

_GenAI Capstone Project · NST Sonipat · February 2026_
