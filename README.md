# Early Warning Student Dropout Risk Prediction

End-to-end machine learning project (capstone, Postgraduate Diploma in Artificial Intelligence and Machine Learning) that flags students likely to drop out **at the end of Semester 2**, early enough for academic advisors to intervene. It covers problem framing, data preparation, modelling, explainability, a fairness audit with evaluated mitigation, and two presentations (technical and business).

## Key results (hold-out set, 885 students, 284 actual dropouts)

| Model (ranked by 5-fold CV F2) | CV F2 | Recall | Precision | F2 | ROC-AUC |
|---|---|---|---|---|---|
| **XGBoost (selected)** | 0.825 | 0.912 | 0.685 | 0.855 | 0.937 |
| Logistic Regression | 0.806 | 0.835 | 0.780 | 0.823 | 0.928 |
| LightGBM | 0.803 | 0.852 | 0.783 | 0.837 | 0.935 |
| Random Forest | 0.793 | 0.835 | 0.785 | 0.824 | 0.929 |

- **Goal:** recall of at least 85% on dropouts (missing a future dropout costs far more than an unnecessary advisor check-in). Models are tuned and selected on cross-validated F2 using the training set only; the test set is used for reporting.
- **Operating point:** XGBoost at threshold 0.50 catches 259 of 284 dropouts (25 missed) and raises 119 unnecessary check-ins, flagging 42.7% of students.
- **Top drivers (SHAP):** Semester 2 approval rate, Semester 1 approval rate, tuition fees up to date, age at enrollment, course. Adding Semester 2 data lifts ROC-AUC from 0.912 to 0.937.
- **Fairness:** five attributes were audited. Recall gaps above 0.10 were found for age group (0.18) and scholarship status (0.13). Group-specific thresholds for age group reduced that gap to 0.07, at the cost of recall (0.912 to 0.898) and precision (0.685 to 0.614). The scholarship gap is still open. Gender and marital status are excluded from the model inputs and used only for auditing; nationality and international-student status remain model inputs and have not been audited.

## Approach

| Step | What is done |
|---|---|
| 1. Framing | Binary classification (1 = Dropout; Graduate and Enrolled = Retained), decision point at end of Semester 2, asymmetric cost matrix |
| 2. Data | UCI *Predict Students' Dropout and Academic Success*, 4,424 students, 36 features; no missing values or duplicates; data dictionary |
| 3. Preprocessing and features | Outlier screen (IQR, retained), four engineered features, RFECV selection, PCA diagnostics, scaler/RFECV/PCA fitted on training data only |
| 4. Modelling | Logistic Regression, Random Forest, XGBoost, LightGBM tuned by grid search with stratified 5-fold CV on F2 |
| 5. Explainability and ethics | Leakage and overfitting audit, SHAP and LIME, subgroup fairness audit, reweighing and group-threshold mitigation compared on the hold-out set |
| 6. Communication | Technical slides generated from the notebook; business deck in `reports/` |

## Repository structure

```text
.
├── data/
│   ├── raw/             # downloaded automatically by the notebook (not committed)
│   └── processed/
├── notebooks/
│   └── Rex_Villamar_Pillar_5_Capstone_Project.ipynb   # full analysis with executed outputs
├── src/
│   ├── features.py      # target binarisation and feature engineering
│   └── fairness.py      # subgroup fairness audit helpers
├── models/
│   ├── model_comparison.csv
│   └── best_model_config.json   # parameters, threshold, seed, feature list
├── reports/
│   ├── Technical_Slides_Rex_Villamar_Pillar_5_Capstone_Project.pptx   # 12-slide technical deck
│   ├── Business_Slides_Rex_Villamar_Pillar_5_Capstone_Project.pptx    # 12-slide executive briefing
│   └── Technical_Report_Full_Notebook.pdf                             # full executed notebook as PDF
├── tests/
│   └── test_helpers.py
├── requirements.txt
└── README.md
```

The notebook is the source of truth; `src/` holds helper functions extracted from it. Running Step 4 of the notebook also writes the fitted `.joblib` models into `models/`.

## Quickstart

```bash
git clone https://github.com/<your-username>/PILLAR_5_CAPSTONE_PROJECT.git
cd PILLAR_5_CAPSTONE_PROJECT
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/Rex_Villamar_Pillar_5_Capstone_Project.ipynb   # then Run All
```

The notebook downloads the dataset from the UCI repository on first run and uses `random_state=42` throughout. It was developed in Google Colab; the Step 6 export cell and the Step 7 Drive cell use Colab paths and can be skipped elsewhere.

## Limitations

- One institution, one country, one cohort period: results need local validation.
- Students still *Enrolled* are labelled Retained although their outcome is unknown (label censoring).
- About 43% to 47% of students are flagged to catch roughly 90% of dropouts, so advisor capacity is the practical constraint.
- The top models are close in cross-validated F2, so the choice among them is partly a judgement call.
- Dollar figures in the business deck (tuition value, success rate, programme cost) are illustrative planning assumptions, not dataset values.

## Data source

Realinho, V., Machado, J., Baptista, L., Martins, M.V. (2022). *Predicting Student Dropout and Academic Success.* Data, 7(11), 146. Dataset: UCI Machine Learning Repository, *Predict Students' Dropout and Academic Success*.

## Author

Rex Villamar, September 2026.
