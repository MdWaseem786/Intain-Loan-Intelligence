# Phase 3 — Servicer Anomaly Intelligence

- Total servicing records: 10,200
- Synthetic records: 200
- Flagged records: 33


## Rule trigger counts

- balance_discrepancy_flag: 1
- delinquency_status_lag_flag: 32
- missing_modification_flag: 2
- invalid_value_flag: 0

## Severity distribution

- NONE: 10,167
- MEDIUM: 31
- HIGH: 2

## Ground-truth evaluation

- Precision: 0.3636
- Recall: 0.0585
- F1: 0.1008


Confusion matrix:

[[9974, 21], [193, 12]]

## Detection philosophy

The anomaly engine uses deterministic business-rule validation. It does not use machine learning to manufacture anomaly labels. Synthetic anomalies are used only for evaluation.
