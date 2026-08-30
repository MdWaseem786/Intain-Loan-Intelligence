# Phase 2 — ML Model Evaluation Report

- Training rows: 1,371,103
- Test rows: 285,776
- Holdout rows: 285,776
- Numeric features: 28
- Validation: 2016-01 to 2017-12
- Final training cutoff: 2017-12


## Binary targets


### next_3m_delinquency_flag
- Model: HistGradientBoosting
- Threshold: 0.2800
- Validation PR-AUC: 0.6978
- Validation ROC-AUC: 0.9316
- Validation F1: 0.6927
- Validation Brier: 0.0141
- Holdout PR-AUC: 0.7026
- Holdout ROC-AUC: 0.9206
- Holdout F1: 0.6935
- Holdout Brier: 0.0180


### next_6m_delinquency_flag
- Model: HistGradientBoosting
- Threshold: 0.3400
- Validation PR-AUC: 0.6550
- Validation ROC-AUC: 0.9018
- Validation F1: 0.6564
- Validation Brier: 0.0207
- Holdout PR-AUC: 0.6476
- Holdout ROC-AUC: 0.8840
- Holdout F1: 0.6425
- Holdout Brier: 0.0271


### next_12m_default_flag
- Model: HistGradientBoosting
- Threshold: 0.9900
- Validation PR-AUC: 0.2290
- Validation ROC-AUC: 0.7200
- Validation F1: 0.4155
- Validation Brier: 0.0031
- Holdout PR-AUC: 0.0610
- Holdout ROC-AUC: 0.8303
- Holdout F1: 0.0698
- Holdout Brier: 0.0040


### next_12m_prepayment_flag
- Model: HistGradientBoosting
- Threshold: 0.1900
- Validation PR-AUC: 0.2878
- Validation ROC-AUC: 0.6168
- Validation F1: 0.2998
- Validation Brier: 0.1232
- Holdout PR-AUC: 0.4040
- Holdout ROC-AUC: 0.6529
- Holdout F1: 0.3667
- Holdout Brier: 0.1225


## Next state
- Validation macro-F1: 0.3145
- Validation weighted-F1: 0.9594
- Holdout macro-F1: 0.2894
- Holdout weighted-F1: 0.9539


## Leakage controls
- Chronological validation was used.
- 2018–2025 holdout was not used for model selection.
- Target/censoring/future columns were excluded.
- loan_id and monthly_reporting_period were excluded as predictors.


## Artifacts
- Predictions: `C:\antigravity\Intain-Loan-Intelligence\data\processed\ml_predictions_test.csv`
- Comparison: `C:\antigravity\Intain-Loan-Intelligence\reports\model_comparison.csv`
- Models: `models/`
