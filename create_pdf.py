from fpdf import FPDF
from fpdf.enums import XPos, YPos

class PDF(FPDF):
    def header(self):
        # Full header only on page 1; compact on subsequent pages
        if self.page_no() == 1:
            self.set_font('helvetica', 'B', 16)
            self.set_text_color(26, 26, 26)
            self.cell(0, 10, 'Credit Risk Prediction - Technical Report', align='C',
                      new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_font('helvetica', 'I', 10)
            self.set_text_color(100, 100, 100)
            self.cell(0, 6, 'GenAI Capstone Project - NST Sonipat - February 2026', align='C',
                      new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.ln(3)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def section_title(self, label):
        self.set_font('helvetica', 'B', 13)
        self.set_text_color(0, 0, 0)
        self.cell(0, 10, label, align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def sub_heading(self, label):
        self.set_font('helvetica', 'B', 11)
        self.set_text_color(30, 30, 30)
        self.cell(0, 8, label, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def body(self, text):
        self.set_font('helvetica', '', 10)
        self.set_text_color(51, 51, 51)
        clean = text.replace('\u2014', '-').replace('\u2013', '-').replace('\u20b9', 'Rs.').replace('\u2022', '*')
        self.multi_cell(0, 5, clean)
        self.ln(3)

    def bullet(self, text):
        self.set_font('helvetica', '', 10)
        self.set_text_color(51, 51, 51)
        clean = text.replace('\u2014', '-').replace('\u2013', '-')
        x = self.get_x()
        self.cell(6, 5, '-')
        self.multi_cell(0, 5, clean)
        self.ln(1)

    def table_header(self, headers, widths):
        self.set_font('helvetica', 'B', 9)
        self.set_fill_color(235, 235, 235)
        for i, h in enumerate(headers):
            self.cell(widths[i], 7, h, border=1, fill=True, align='C')
        self.ln()

    def table_row(self, row, widths, align='L'):
        self.set_font('helvetica', '', 9)
        for i, d in enumerate(row):
            self.cell(widths[i], 6, str(d), border=1, align=align)
        self.ln()


def generate_pdf():
    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── 1. Problem Statement ──
    pdf.section_title('1. Problem Statement')
    pdf.body("Access to credit is a fundamental enabler of economic participation, yet lending institutions face the persistent challenge of distinguishing creditworthy borrowers from those likely to default. Inaccurate risk assessments lead to significant financial losses through non-performing assets or the denial of credit to deserving applicants.")
    pdf.body("This project addresses the binary classification problem of predicting loan default risk. Given applicant attributes - income, employment history, loan grade, prior default records - the goal is to predict whether a borrower will default (status=1) or repay (status=0).")
    pdf.body("The trained model is deployed as an interactive Streamlit web application for real-time credit risk evaluation.")

    # ── 2. Data Description ──
    pdf.section_title('2. Data Description')
    pdf.body("The dataset contains 32,581 records of historical loan applications.")

    pdf.sub_heading('Features')
    w = [48, 38, 104]
    pdf.table_header(["Feature", "Type", "Description"], w)
    for row in [
        ["person_age", "Numerical", "Age of the applicant"],
        ["person_income", "Numerical", "Annual income"],
        ["person_emp_length", "Numerical", "Employment length (years)"],
        ["person_home_ownership", "Categorical", "RENT, OWN, MORTGAGE, OTHER"],
        ["loan_intent", "Categorical", "PERSONAL, EDUCATION, MEDICAL, etc."],
        ["loan_grade", "Ordinal", "Credit grade (A-G)"],
        ["loan_amnt", "Numerical", "Loan amount requested"],
        ["loan_int_rate", "Numerical", "Interest rate (%)"],
        ["loan_percent_income", "Numerical", "Loan amount as % of income"],
        ["cb_person_default", "Binary", "Previous default on record (Y/N)"],
        ["cb_person_cred_hist", "Numerical", "Credit history length (years)"],
        ["loan_status (Target)", "Binary", "0 = Non-Default, 1 = Default"],
    ]:
        pdf.table_row(row, w)
    pdf.ln(3)
    pdf.body("Class Distribution: Non-Default ~78% | Default ~22%")

    # ── 3. EDA ──
    pdf.section_title('3. Exploratory Data Analysis (EDA)')
    pdf.bullet("Target Distribution: Confirmed class imbalance; addressed via balanced weighting.")
    pdf.bullet("Income vs. Default: Defaulters tend to have lower incomes.")
    pdf.bullet("Loan-to-Income Ratio: Strongest class separator; high ratios = higher risk.")
    pdf.bullet("Interest Rate: Defaulters concentrated in higher rate brackets.")
    pdf.bullet("Loan Grade: Default rates increase progressively from A to G.")
    pdf.bullet("Correlation: loan_percent_income has strongest positive correlation with default.")

    # ── 4. Methodology ──
    pdf.section_title('4. Methodology')
    pdf.sub_heading('4.1 Preprocessing')
    pdf.body("1. Missing values: Median imputation (emp_length, int_rate)\n2. Ordinal encoding: Loan grade A-G mapped to 1-7\n3. Binary encoding: Default history Y/N to 1/0\n4. One-hot encoding: Home ownership, loan intent (drop_first)\n5. Feature scaling: StandardScaler normalization")
    pdf.sub_heading('4.2 Train-Test Split')
    pdf.body("75% training / 25% testing (random_state=42)")
    pdf.sub_heading('4.3 Models Trained')
    pdf.body("Model 1: Logistic Regression (baseline, max_iter=1000)\nModel 2: Decision Tree Classifier (selected) - max_depth=10, min_samples_split=20, min_samples_leaf=10, class_weight='balanced'")
    pdf.body("Rationale: Decision Tree selected for higher accuracy (90.8% vs 84.5%) and better recall on the default class (0.77), which is critical in credit risk assessment.")

    # ── 5. Evaluation ──
    pdf.section_title('5. Evaluation')
    pdf.sub_heading('5.1 Model Comparison')
    w3 = [60, 60, 60]
    pdf.table_header(["Metric", "Logistic Regression", "Decision Tree"], w3)
    pdf.table_row(["Accuracy", "84.52%", "90.81%"], w3, 'C')
    pdf.table_row(["ROC-AUC", "0.8636", "0.6817"], w3, 'C')
    pdf.ln(3)

    pdf.sub_heading('5.2 Classification Report (Decision Tree)')
    w5 = [35, 35, 35, 35, 35]
    pdf.table_header(["Class", "Precision", "Recall", "F1-Score", "Support"], w5)
    pdf.table_row(["Non-Default (0)", "0.93", "0.95", "0.94", "6331"], w5, 'C')
    pdf.table_row(["Default (1)", "0.81", "0.77", "0.79", "1815"], w5, 'C')
    pdf.table_row(["Weighted Avg", "0.91", "0.91", "0.91", "8146"], w5, 'C')
    pdf.ln(3)

    pdf.sub_heading('5.3 Confusion Matrix')
    pdf.table_header(["Actual / Predicted", "Non-Default", "Default"], w3)
    pdf.table_row(["Non-Default", "6,006 (TN)", "325 (FP)"], w3, 'C')
    pdf.table_row(["Default", "424 (FN)", "1,391 (TP)"], w3, 'C')
    pdf.ln(3)

    # ── 6. Optimization & Limitations ──
    pdf.section_title('6. Optimization & Limitations')
    pdf.bullet("Class Imbalance: Used class_weight='balanced' to upweight minority class.")
    pdf.bullet("Tree Pruning: max_depth=10 prevents overfitting while maintaining generalization.")
    pdf.bullet("Limitation: Binary default history; cannot capture partial default rates.")
    pdf.bullet("Limitation: ROC-AUC of 0.68 indicates room for probability calibration improvement.")

    # ── 7. Deployment ──
    pdf.section_title('7. Deployment')
    pdf.body("Live App: https://genaicapstone-a7eipdbqudn2niewt9s2mp.streamlit.app/")
    pdf.bullet("Real-time prediction via interactive form.")
    pdf.bullet("Three-tier decision: Approved (<20%), Needs Review (20-50%), Rejected (>50%).")
    pdf.bullet("Displays default probability, repayment likelihood, and loan grade.")

    # ── 8. Team Contribution ──
    pdf.section_title('8. Team Contribution')
    wt = [50, 140]
    pdf.table_header(["Member", "Contribution"], wt)
    pdf.table_row(["Palak", "Data preprocessing, EDA, model training & evaluation"], wt)
    pdf.table_row(["Samarth", "Streamlit app, UI/UX design, deployment, bug fixes"], wt)
    pdf.ln(3)

    # ── 9. Tech Stack ──
    pdf.section_title('9. Tech Stack')
    wts = [50, 140]
    pdf.table_header(["Component", "Technology"], wts)
    for row in [
        ["Language", "Python 3.x"],
        ["ML Libraries", "scikit-learn, pandas, NumPy"],
        ["Visualization", "Matplotlib, Seaborn"],
        ["Web Framework", "Streamlit"],
        ["Deployment", "Streamlit Cloud"],
        ["Version Control", "Git & GitHub"],
    ]:
        pdf.table_row(row, wts)
    pdf.ln(3)

    # ── 10. System Architecture ──
    pdf.section_title('10. System Architecture')
    pdf.body("The system is structured into three functional layers:")
    pdf.bullet("Layer 1 - User Interface (Streamlit): Handles applicant forms and input validation.")
    pdf.bullet("Layer 2 - Intelligence (Python/Scikit-Learn): Manages preprocessing, scaling, and Decision Tree model logic.")
    pdf.bullet("Layer 3 - Output Display: Visualizes risk metrics, tier status, and probability scores.")
    pdf.ln(2)
    pdf.sub_heading("The Data Journey:")
    pdf.body("1. Input: User provides details (Income, Age, etc.) via the web form.\n2. Transformation: Data is cleaned and scaled for the model.\n3. Prediction: Decision Tree calculates default probability.\n4. Action: Result is categorized and displayed on the dashboard.")

    # ── 11. Input-Output Specification ──
    pdf.section_title('11. Input-Output Specification')
    pdf.sub_heading("System Inputs (Borrower Profile)")
    wio = [38, 58, 94]
    pdf.table_header(["Category", "Key Features", "Purpose"], wio)
    pdf.table_row(["Demographics", "Age, Emp Length", "Assess stability and life-stage"], wio)
    pdf.table_row(["Financials", "Income, Loan Amount", "Calculate Loan-to-Income ratio"], wio)
    pdf.table_row(["History", "Default/Credit Hist", "Factor in past behavior"], wio)
    pdf.table_row(["Context", "Home/Loan Purpose", "Collateral type and context"], wio)
    pdf.ln(3)

    pdf.sub_heading("System Outputs (Credit Decision)")
    pdf.table_header(["Result Type", "Output Detail", "Description"], wio)
    pdf.table_row(["Score", "Default Prob %", "Statistical risk of non-repayment"], wio)
    pdf.table_row(["Decision", "Status Tier", "Approved / Review / Rejected"], wio)
    pdf.table_row(["Metrics", "Repayment %", "Confidence in repayment"], wio)
    pdf.ln(3)

    # ── 12. Key Risk Drivers ──
    pdf.section_title('12. Key Risk Drivers (Ranked by Impact)')
    pdf.body("The following features are the primary drivers of credit risk, ranked by impact:")
    pdf.bullet("1. Loan-to-Income Ratio: Single biggest predictor; high debt is a primary rejection trigger.")
    pdf.bullet("2. Interest Rate: Strong signal of pre-existing risk profiles assigned by lenders.")
    pdf.bullet("3. Loan Grade: Summary quality metric; Grades E-G carry heavy penalties.")
    pdf.bullet("4. Annual Income: Baseline affordability check; low income keeps risk high.")

    # ── 13. References ──
    pdf.section_title('13. References')
    pdf.body("1. scikit-learn Documentation - https://scikit-learn.org/stable/\n2. Streamlit Documentation - https://docs.streamlit.io/\n3. Credit Risk Dataset - Hosted on GitHub repository\n4. Decision Tree Classifier - Breiman, L. (1984). Classification and Regression Trees")

    pdf.output("REPORT.pdf")
    print("REPORT.pdf generated successfully (sections 1-13).")

if __name__ == "__main__":
    generate_pdf()
