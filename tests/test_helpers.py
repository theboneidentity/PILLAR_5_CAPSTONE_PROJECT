import numpy as np
import pandas as pd
from src.features import binarize_target, engineer_features, ENGINEERED
from src.fairness import build_groups, group_metrics, fairness_summary


def _toy():
    rng = np.random.default_rng(0)
    n = 400
    return pd.DataFrame({
        'Curricular units 1st sem (approved)': rng.integers(0, 7, n), 'Curricular units 1st sem (enrolled)': rng.integers(1, 8, n),
        'Curricular units 2nd sem (approved)': rng.integers(0, 7, n), 'Curricular units 2nd sem (enrolled)': rng.integers(1, 8, n),
        'Curricular units 1st sem (grade)': rng.uniform(0, 18, n), 'Curricular units 2nd sem (grade)': rng.uniform(0, 18, n),
        'Debtor': rng.integers(0, 2, n), 'Tuition fees up to date': rng.integers(0, 2, n), 'Scholarship holder': rng.integers(0, 2, n),
        'Gender': rng.integers(0, 2, n), 'Marital status': rng.choice([1, 2, 4], n), 'Age at enrollment': rng.integers(17, 45, n),
        'Target': rng.choice(['Dropout', 'Graduate', 'Enrolled'], n),
    })


def test_features():
    df = engineer_features(binarize_target(_toy()))
    assert set(ENGINEERED) <= set(df.columns) and set(df['Target_Binary']) <= {0, 1}
    assert df['Financial_Stress_Index'].between(0, 3).all()


def test_fairness_helpers():
    df = engineer_features(binarize_target(_toy()))
    y = df['Target_Binary'].values
    gm = group_metrics(build_groups(df), y, y)          # perfect predictions -> recall 1.0 everywhere
    assert gm['Recall (TPR)'].dropna().eq(1.0).all()
    assert (fairness_summary(gm)['TPR gap'] == 0).all()
