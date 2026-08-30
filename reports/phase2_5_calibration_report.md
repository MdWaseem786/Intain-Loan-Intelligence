# 12-Month Default Probability Calibration

## Validation

- Raw Brier: 0.001427
- Sigmoid Brier: 0.001435
- Isotonic Brier: 0.001186

- Raw Log Loss: 0.013332
- Sigmoid Log Loss: 0.008638
- Isotonic Log Loss: 0.005741

Selected calibration: **isotonic**

## Holdout

- Raw Brier: 0.004002
- Calibrated Brier: 0.002589
- Raw Log Loss: 0.038652
- Calibrated Log Loss: 0.016596
- Raw ROC-AUC: 0.830279
- Calibrated ROC-AUC: 0.850156
- Raw PR-AUC: 0.060955
- Calibrated PR-AUC: 0.057890