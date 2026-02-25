import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Credit Risk Evaluator",
    page_icon="📋",
    layout="centered"
)

# ── Minimal clean styling ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #f7f7f5;
    color: #1a1a1a;
}

.block-container { padding: 2.5rem 2rem; max-width: 720px; }

h1 { font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem; font-weight: 500; letter-spacing: -0.5px; margin-bottom: 0; }
h3 { font-size: 0.8rem; font-weight: 400; color: #888; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 2rem; margin-bottom: 0.5rem; border-bottom: 1px solid #e0e0e0; padding-bottom: 6px; }

.stButton > button {
    background: #1a1a1a;
    color: #f7f7f5;
    border: none;
    border-radius: 4px;
    padding: 0.6rem 2rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 0.5px;
    width: 100%;
    margin-top: 1rem;
    cursor: pointer;
    transition: background 0.15s;
}
.stButton > button:hover { background: #333; }

.stSelectbox > div > div, .stNumberInput > div > div > input {
    border: 1px solid #ddd !important;
    border-radius: 4px !important;
    background: #fff !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.9rem !important;
}

.result-box {
    margin-top: 2rem;
    padding: 1.5rem;
    border-radius: 6px;
    text-align: center;
    border: 1.5px solid;
}
.approved   { background: #f0faf0; border-color: #4caf50; }
.review     { background: #fffbf0; border-color: #ff9800; }
.rejected   { background: #fff0f0; border-color: #f44336; }

.result-label { font-family: 'IBM Plex Mono', monospace; font-size: 1.1rem; font-weight: 500; margin-bottom: 4px; }
.result-prob  { font-size: 0.85rem; color: #666; }

.metric-row { display: flex; gap: 1rem; margin-top: 1rem; }
.metric-box { flex: 1; background: #fff; border: 1px solid #e0e0e0; border-radius: 6px; padding: 1rem; text-align: center; }
.metric-val { font-family: 'IBM Plex Mono', monospace; font-size: 1.3rem; font-weight: 500; }
.metric-lbl { font-size: 0.75rem; color: #888; margin-top: 2px; }

.divider { height: 1px; background: #e0e0e0; margin: 1.5rem 0; }
.note { font-size: 0.78rem; color: #aaa; margin-top: 0.5rem; }

footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Train model (cached so it only runs once) ──────────────────────────────────
@st.cache_resource
def load_model():
    url = "https://raw.githubusercontent.com/PALAK7890/GenAI_Capstone/refs/heads/main/credit_risk_dataset.csv"
    df = pd.read_csv(url)

    df['person_emp_length'].fillna(df['person_emp_length'].median(), inplace=True)
    df['loan_int_rate'].fillna(df['loan_int_rate'].median(), inplace=True)

    grade_map = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7}
    df['loan_grade'] = df['loan_grade'].map(grade_map)
    df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map({'Y':1,'N':0})
    df = pd.get_dummies(df, columns=['person_home_ownership','loan_intent'], drop_first=True)
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = DecisionTreeClassifier(
        max_depth=10, min_samples_split=20,
        min_samples_leaf=10, class_weight='balanced', random_state=42
    )
    model.fit(X_train_sc, y_train)

    return model, scaler, list(X.columns)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# Credit Risk Evaluator")
st.markdown("<p style='color:#888; font-size:0.9rem; margin-top:4px;'>Decision Tree · GenAI Capstone · NST Sonipat</p>", unsafe_allow_html=True)
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
with st.spinner("Loading model..."):
    model, scaler, feature_cols = load_model()

# ── Form ───────────────────────────────────────────────────────────────────────
st.markdown("### Applicant Info")
col1, col2 = st.columns(2)
with col1:
    age    = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Annual Income (₹)", min_value=0, value=500000, step=10000)
with col2:
    emp_len   = st.number_input("Employment Length (years)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
    home_own  = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])

st.markdown("### Loan Details")
col3, col4 = st.columns(2)
with col3:
    loan_amnt   = st.number_input("Loan Amount (₹)", min_value=500, max_value=3500000, value=150000, step=5000)
    loan_intent = st.selectbox("Loan Purpose", ["PERSONAL","EDUCATION","MEDICAL","VENTURE","HOMEIMPROVEMENT","DEBTCONSOLIDATION"])
with col4:
    loan_grade  = st.selectbox("Loan Grade", ["A","B","C","D","E","F","G"])
    int_rate    = st.number_input("Interest Rate (%)", min_value=1.0, max_value=30.0, value=12.5, step=0.1)

st.markdown("### Credit History")
col5, col6 = st.columns(2)
with col5:
    cred_hist = st.number_input("Credit History Length (years)", min_value=0, max_value=50, value=6)
with col6:
    default_file = st.selectbox("Previous Default on File?", ["No", "Yes"])

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ── Predict ────────────────────────────────────────────────────────────────────
if st.button("Evaluate Risk"):

    grade_map = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7}
    pct_income = loan_amnt / income if income > 0 else 0

    # Build input matching training columns
    raw = {
        "person_age": age,
        "person_income": income,
        "person_emp_length": emp_len,
        "loan_grade": grade_map[loan_grade],
        "loan_amnt": loan_amnt,
        "loan_int_rate": int_rate,
        "loan_percent_income": pct_income,
        "cb_person_default_on_file": 1 if default_file == "Yes" else 0,
        "cb_person_cred_hist_length": cred_hist,
        "person_home_ownership_OTHER":    1 if home_own == "OTHER" else 0,
        "person_home_ownership_OWN":      1 if home_own == "OWN" else 0,
        "person_home_ownership_RENT":     1 if home_own == "RENT" else 0,
        "loan_intent_EDUCATION":          1 if loan_intent == "EDUCATION" else 0,
        "loan_intent_HOMEIMPROVEMENT":    1 if loan_intent == "HOMEIMPROVEMENT" else 0,
        "loan_intent_MEDICAL":            1 if loan_intent == "MEDICAL" else 0,
        "loan_intent_PERSONAL":           1 if loan_intent == "PERSONAL" else 0,
        "loan_intent_VENTURE":            1 if loan_intent == "VENTURE" else 0,
    }

    # Align columns with training set
    input_df = pd.DataFrame([raw])
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_cols]

    input_sc = scaler.transform(input_df)
    prob     = model.predict_proba(input_sc)[0][1]
    risk_pct = round(prob * 100, 1)

    if prob < 0.2:
        decision, css, emoji = "Approved", "approved", "✓"
    elif prob < 0.5:
        decision, css, emoji = "Needs Review", "review", "⚠"
    else:
        decision, css, emoji = "Rejected", "rejected", "✕"

    st.markdown(f"""
    <div class='result-box {css}'>
        <div class='result-label'>{emoji} &nbsp; {decision}</div>
        <div class='result-prob'>Default probability: {risk_pct}%</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class='metric-row'>
        <div class='metric-box'>
            <div class='metric-val'>{risk_pct}%</div>
            <div class='metric-lbl'>Default Risk</div>
        </div>
        <div class='metric-box'>
            <div class='metric-val'>{round(100 - risk_pct, 1)}%</div>
            <div class='metric-lbl'>Repayment Likelihood</div>
        </div>
        <div class='metric-box'>
            <div class='metric-val'>{loan_grade}</div>
            <div class='metric-lbl'>Loan Grade</div>
        </div>
    </div>
    <p class='note'>Model: Decision Tree · Accuracy 90.8% · ROC-AUC 0.68 · Trained on 32,581 records</p>
    """, unsafe_allow_html=True)