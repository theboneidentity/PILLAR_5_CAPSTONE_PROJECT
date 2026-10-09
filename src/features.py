"""Target binarisation and domain feature engineering (notebook Step 3)."""
import pandas as pd

ENGINEERED = ['Approval_Rate_Sem1', 'Approval_Rate_Sem2', 'Grade_Trend', 'Financial_Stress_Index']
SENSITIVE_COLS = ['Gender', 'Marital status']   # withheld from model inputs, used only for the fairness audit


def binarize_target(df: pd.DataFrame) -> pd.DataFrame:
    """1 = Dropout, 0 = Retained (Graduate + Enrolled)."""
    df = df.copy()
    df['Target_Binary'] = df['Target'].apply(lambda x: 1 if x == 'Dropout' else 0)
    return df.drop(columns=['Target'])


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the four domain-derived features (row-wise, so no train/test leakage)."""
    df = df.copy()
    df['Approval_Rate_Sem1'] = df['Curricular units 1st sem (approved)'] / (df['Curricular units 1st sem (enrolled)'] + 1e-5)
    df['Approval_Rate_Sem2'] = df['Curricular units 2nd sem (approved)'] / (df['Curricular units 2nd sem (enrolled)'] + 1e-5)
    df['Grade_Trend'] = df['Curricular units 2nd sem (grade)'] - df['Curricular units 1st sem (grade)']
    df['Financial_Stress_Index'] = df['Debtor'] + (1 - df['Tuition fees up to date']) + (1 - df['Scholarship holder'])
    return df
