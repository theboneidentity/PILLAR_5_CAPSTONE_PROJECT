"""Subgroup fairness audit helpers (notebook Step 5c)."""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

MIN_GROUP_N = 30   # groups smaller than this are too small for stable rates


def build_groups(A: pd.DataFrame) -> pd.DataFrame:
    """Audit groups from the audit table. UCI coding: Gender 1 = male, 0 = female; Marital status 1 = single."""
    g = pd.DataFrame(index=A.index)
    g['Gender'] = A['Gender'].map({0: 'Female', 1: 'Male'})
    g['Marital status'] = np.where(A['Marital status'] == 1, 'Single', 'Not single')
    g['Scholarship holder'] = A['Scholarship holder'].map({0: 'No scholarship', 1: 'Scholarship'})
    g['Debtor'] = A['Debtor'].map({0: 'Not debtor', 1: 'Debtor'})
    g['Age group'] = pd.cut(A['Age at enrollment'], bins=[0, 20, 25, 200], labels=['<=20', '21-25', '>25']).astype(str)
    return g.reset_index(drop=True)


def group_metrics(groups: pd.DataFrame, y_true, y_hat) -> pd.DataFrame:
    """Selection rate, recall (TPR), FPR and precision per group."""
    y_true, y_hat = np.asarray(y_true), np.asarray(y_hat)
    rows = []
    for attr in groups.columns:
        for level in sorted(groups[attr].dropna().unique()):
            mask = (groups[attr] == level).values
            yt, yp = y_true[mask], y_hat[mask]
            tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
            rows.append({
                'Attribute': attr, 'Group': level, 'N': int(mask.sum()),
                'Dropout rate': yt.mean(), 'Selection rate': yp.mean(),
                'Recall (TPR)': tp / (tp + fn) if (tp + fn) else np.nan,
                'FPR': fp / (fp + tn) if (fp + tn) else np.nan,
                'Precision': tp / (tp + fp) if (tp + fp) else np.nan,
            })
    return pd.DataFrame(rows)


def fairness_summary(gm: pd.DataFrame) -> pd.DataFrame:
    """Disparate impact, recall (equal-opportunity) gap, FPR gap and equalized-odds gap per attribute."""
    out = []
    for attr, sub in gm.groupby('Attribute', sort=False):
        sub = sub[sub['N'] >= MIN_GROUP_N]
        if len(sub) < 2:
            continue
        max_sel = sub['Selection rate'].max()
        tpr_gap = sub['Recall (TPR)'].max() - sub['Recall (TPR)'].min()
        fpr_gap = sub['FPR'].max() - sub['FPR'].min()
        out.append({
            'Attribute': attr,
            'Disparate impact': sub['Selection rate'].min() / max_sel if max_sel > 0 else np.nan,
            'TPR gap': tpr_gap, 'FPR gap': fpr_gap,
            'Equalized-odds gap': max(tpr_gap, fpr_gap),
        })
    return pd.DataFrame(out)
