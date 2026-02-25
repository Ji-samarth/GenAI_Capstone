from fpdf import FPDF
from fpdf.enums import XPos, YPos

class PDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 16)
        self.set_text_color(26, 26, 26)
        self.cell(0, 10, 'Credit Risk Prediction - Technical Report', align='C', 
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font('helvetica', 'I', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, 'GenAI Capstone Project - NST Sonipat - February 2026', align='C',
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def chapter_title(self, label):
        self.set_font('helvetica', 'B', 14)
        self.set_text_color(0, 0, 0)
        self.cell(0, 10, label, align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def chapter_body(self, body):
        self.set_font('helvetica', '', 11)
        self.set_text_color(51, 51, 51)
        # Handle unicode dashes and symbols
        clean_body = body.replace('—', '-').replace('–', '-').replace('₹', 'Rs.').replace('•', '*')
        self.multi_cell(0, 6, clean_body)
        self.ln(4)

    def bullet_point(self, text):
        self.set_font('helvetica', '', 11)
        self.set_text_color(51, 51, 51)
        self.cell(10, 6, "- ", border=0)
        clean_text = text.replace('—', '-').replace('–', '-').replace('₹', 'Rs.').replace('•', '*')
        self.multi_cell(0, 6, clean_text)
        self.ln(2)

    def table_header(self, headers, widths):
        self.set_font('helvetica', 'B', 10)
        self.set_fill_color(240, 240, 240)
        for i, header in enumerate(headers):
            clean_h = header.replace('—', '-').replace('–', '-').replace('\\', '/')
            self.cell(widths[i], 8, clean_h, border=1, fill=True, align='C')
        self.ln()

    def table_row(self, row, widths, align='L'):
        self.set_font('helvetica', '', 9)
        max_h = 6
        for i, data in enumerate(row):
            clean_d = str(data).replace('—', '-').replace('–', '-').replace('₹', 'Rs.')
            self.cell(widths[i], max_h, clean_d, border=1, align=align)
        self.ln()

def generate_full_pdf():
    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # 1. Problem Statement
    pdf.chapter_title('1. Problem Statement')
    pdf.chapter_body("Access to credit is a fundamental enabler of economic participation, yet lending institutions face the persistent challenge of distinguishing creditworthy borrowers from those likely to default. Inaccurate risk assessments lead to significant financial losses through non-performing assets or, conversely, the denial of credit to deserving applicants.")
    pdf.chapter_body("This project addresses the binary classification problem of predicting loan default risk. Given a set of applicant and loan attributes - such as income, employment history, loan grade, and prior default records - the goal is to build a supervised machine learning model that can predict whether a borrower will default on a loan (loan_status = 1) or repay successfully (loan_status = 0).")
    pdf.chapter_body("The trained model is deployed as an interactive Streamlit web application that enables real-time credit risk evaluation for new applicants.")

    # 2. Data Description
    pdf.chapter_title('2. Data Description')
    pdf.chapter_body("The dataset is a publicly available credit risk dataset hosted on GitHub, containing 32,581 records of historical loan applications.")
    
    pdf.set_font('helvetica', 'B', 11)
    pdf.cell(0, 8, "2.2 Features", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    
    widths = [50, 40, 100]
    headers = ["Feature", "Type", "Description"]
    pdf.table_header(headers, widths)
    
    features = [
        ["person_age", "Numerical", "Age of the applicant"],
        ["person_income", "Numerical", "Annual income of the applicant"],
        ["person_emp_length", "Numerical", "Employment length in years"],
        ["person_home_ownership", "Categorical", "Home ownership status (RENT, OWN, ...)"],
        ["loan_intent", "Categorical", "Purpose of loan (PERSONAL, EDUCATION, ...)"],
        ["loan_grade", "Categorical", "Credit grade assigned (A-G)"],
        ["loan_amnt", "Numerical", "Loan amount requested"],
        ["loan_int_rate", "Numerical", "Interest rate on the loan"],
        ["loan_percent_income", "Numerical", "Loan amount as % of annual income"],
        ["cb_person_default", "Binary", "Previous default on record (Y/N)"],
        ["cb_person_cred_hist", "Numerical", "Length of credit history"],
        ["loan_status", "Binary", "Target (0 = Non-Default, 1 = Default)"]
    ]
    for row in features:
        pdf.table_row(row, widths)
    pdf.ln(5)

    pdf.chapter_body("Class Distribution: Non-Default (0): ~78% | Default (1): ~22%")

    # 3. EDA
    pdf.chapter_title('3. Exploratory Data Analysis (EDA)')
    pdf.bullet_point("Target Distribution: Confirmed class imbalance (addressed via balanced weighting).")
    pdf.bullet_point("Income vs. Default: Defaulters tend to have lower incomes.")
    pdf.bullet_point("Loan Amount vs. Default: Higher loan amounts associate with increased risk.")
    pdf.bullet_point("Loan-to-Income Ratio: Strongest separator; high ratios indicate higher risk.")
    pdf.bullet_point("Interest Rate vs. Default: Riskier borrowers are assigned higher rates.")
    pdf.bullet_point("Correlation: Loan percent income has the strongest positive correlation with default.")

    # 4. Methodology
    pdf.chapter_title('4. Methodology')
    pdf.chapter_body("4.1 Preprocessing: Median imputation for missing values in employment length and interest rate; Ordinal encoding for grades; One-hot encoding for categorical features; StandardScaler normalization.")
    pdf.chapter_body("4.2 Train-Test Split: 75% Training, 25% Testing (random_state=42).")
    pdf.chapter_body("4.3 Model: Decision Tree Classifier (max_depth=10, min_samples_split=20, min_samples_leaf=10, class_weight='balanced').")
    pdf.chapter_body("Rationale: Decision Tree was selected over Logistic Regression for its higher overall accuracy (90.8%) and superior recall for the default class (0.77).")

    # 5. Evaluation
    pdf.chapter_title('5. Evaluation')
    pdf.set_font('helvetica', 'B', 11)
    pdf.cell(0, 8, "5.1 Model Comparison", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.table_header(["Metric", "Logistic Regression", "Decision Tree"], [60, 60, 60])
    pdf.table_row(["Accuracy", "84.52%", "90.81%"], [60, 60, 60], align='C')
    pdf.table_row(["ROC-AUC", "0.8636", "0.6817"], [60, 60, 60], align='C')
    pdf.ln(5)

    pdf.set_font('helvetica', 'B', 11)
    pdf.cell(0, 8, "5.2 Classification Report (Decision Tree)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.table_header(["Class", "Precision", "Recall", "F1-Score", "Support"], [35, 35, 35, 35, 35])
    pdf.table_row(["Non-Default (0)", "0.93", "0.95", "0.94", "6331"], [35, 35, 35, 35, 35], align='C')
    pdf.table_row(["Default (1)", "0.81", "0.77", "0.79", "1815"], [35, 35, 35, 35, 35], align='C')
    pdf.table_row(["Weighted Avg", "0.91", "0.91", "0.91", "8146"], [35, 35, 35, 35, 35], align='C')
    pdf.ln(5)

    pdf.set_font('helvetica', 'B', 11)
    pdf.cell(0, 8, "5.3 Confusion Matrix", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.table_header(["Actual / Predicted", "Non-Default", "Default"], [60, 60, 60])
    pdf.table_row(["Non-Default", "6,006 (TN)", "325 (FP)"], [60, 60, 60], align='C')
    pdf.table_row(["Default", "424 (FN)", "1,391 (TP)"], [60, 60, 60], align='C')
    pdf.ln(5)

    # 6. Optimization & Limitations
    pdf.chapter_title('6. Optimization & Limitations')
    pdf.bullet_point("Class Imbalance: used balanced weighting.")
    pdf.bullet_point("Pruning: max_depth=10 to prevent overfitting.")
    pdf.bullet_point("Limitations: Binary default history misses partial defaults; limited probability calibration (ROC-AUC 0.68).")

    # 7. Deployment
    pdf.chapter_title('7. Deployment')
    pdf.chapter_body("Streamlit App: https://genaicapstone-a7eipdbqudn2niewt9s2mp.streamlit.app/")

    # 8. Team
    pdf.chapter_title('8. Team Contribution')
    pdf.table_header(["Team Member", "Contribution"], [60, 130])
    pdf.table_row(["Palak", "Data preprocessing, EDA, model training & evaluation"], [60, 130])
    pdf.table_row(["Samarth", "App development, UI/UX design, deployment, bug fixes"], [60, 130])

    pdf.output("REPORT.pdf")
    print("Full REPORT.pdf generated successfully.")

if __name__ == "__main__":
    generate_full_pdf()
