# Phase 1B: Target Engineering & Data Pack Report

**Generated**: 2026-08-26 18:18:13
**Runtime**: 43.0 seconds
**Reproduction**: `python scripts/build_phase1b.py`

---

## 1. Dataset Summary

| Metric | Value |
|---|---|
| Source | `data/processed/selected_loan_performance.parquet` |
| Total unique loans | 25,000 |
| Total monthly observations | 1,660,802 |
| Actual date range | 07/2009 – 03/2026 |
| Train observations (≤ 2017-12) | 1,371,103 |
| Test observations (2018-01 – 2025-12) | 285,776 |
| Train unique loans | 25,000 |
| Test unique loans | 5,591 |
| Non-consecutive month gaps | 2 |

---

## 2. Temporal Boundaries

| Period | Start | End | Purpose |
|---|---|---|---|
| **Train** | Earliest available (07/2009) | 2017-12 | Model training with targets |
| **Test** | 2018-01 | 2025-12 | Prediction-time features only |
| **Support** | 2026-01 | 2026-03 | Target construction windows only |

2026 observations are used exclusively for computing forward-looking targets
near the dataset boundary. They do not appear as standalone train/test rows.

---

## 3. Target Definitions

### 3.1 next_3m_delinquency_flag
- **Window**: T+1 through T+3
- **Positive**: Any month has `current_delinquency_status` ≥ '01' (30+ DPD)
- **XX handling**: 'XX' does NOT count as delinquency
- **Train rate**: 1.81% (24,524 / 1,351,694)

### 3.2 next_6m_delinquency_flag
- **Window**: T+1 through T+6
- **Positive**: Any month has `current_delinquency_status` ≥ '01'
- **Train rate**: 2.53% (34,216 / 1,351,694)

### 3.3 next_12m_default_flag
- **Window**: T+1 through T+12
- **Positive**: `zero_balance_removal_reason` ∈ {02, 03, 09}
  - 02 = Third-Party Sale (foreclosure)
  - 03 = Short Sale
  - 09 = REO Disposition
- **NOT default**: Voluntary prepayment (01), Repurchase (06), Note Sale (15), Reperforming Sale (16)
- **NOT default**: 90+ DPD alone (this is a risk indicator, not a termination event)
- **Train rate**: 0.13% (1,738 / 1,351,694)

### 3.4 next_12m_prepayment_flag
- **Window**: T+1 through T+12
- **Positive**: `zero_balance_removal_reason` = '01' (voluntary payoff/matured)
- **NOT prepayment**: Any other termination reason
- **Train rate**: 16.93% (228,826 / 1,351,694)

### 3.5 next_state
- **Window**: T+1 (single step)
- **Taxonomy**: Current, 30DPD, 60DPD, 90+DPD, Prepaid, Default/REO, Unknown

---

## 4. Next-State Distribution (Train)

| State | Count |
|---|---|
| Current | 1,316,318 |
| NaN (censored/last obs) | 19,409 |
| Prepaid | 19,287 |
| 30DPD | 8,355 |
| 90+DPD | 5,682 |
| 60DPD | 1,866 |
| Default/REO | 143 |
| Unknown | 43 |

---

## 5. Right-Censoring

| Window | Censored (train) | Description |
|---|---|---|
| 3-month | 19,409 | Loan active, <3 future months in dataset |
| 6-month | 19,409 | Loan active, <6 future months in dataset |
| 12-month | 19,409 | Loan active, <12 future months in dataset |

**Rule**: A target is available if the loan terminated (fate is known) OR
enough future calendar months exist in the dataset. Otherwise, the
observation is right-censored and the target is NaN.

Terminated loans always have available targets regardless of remaining
months, because the loan's outcome is fully observed.

---

## 6. XX / Unknown Status Handling

- `XX` = Unknown delinquency status (often forbearance-related)
- XX does **NOT** count as a positive delinquency event
- XX is **NOT** treated as Current ('00')
- XX is mapped to state `Unknown` in the state taxonomy
- Separate tracking flags: `unknown_status_present_3m/6m/12m`

| Window | Observations with XX in window (train) |
|---|---|
| 3-month | 58,508 |
| 6-month | 116,983 |
| 12-month | 231,070 |

---

## 7. Feature Engineering

All features use information available at or before prediction time T.

| Feature | Source | Rolling? |
|---|---|---|
| `pct_life_elapsed` | loan_age / original_loan_term | No |
| `rate_spread` | current_rate - original_rate | No |
| `upb_pct_original` | current_upb / original_upb | No |
| `ever_delinquent` | Cumulative max of delinq flag | Yes (cumulative) |
| `max_delinquency_to_date` | Cumulative max of delinq months | Yes (cumulative) |
| `months_since_last_delinquency` | Months since last 30+ DPD | Yes (cumulative) |
| `delinquency_count_12m` | Rolling 12-month delinq count | Yes (12m window) |
| `ever_modified` | Cumulative max of mod flag | Yes (cumulative) |

---

## 8. Critical Leakage Audit

### 8.1 Methodology

For every observation at time T, the following separation is enforced:

| Category | Allowed Sources | Example |
|---|---|---|
| **Features** | Origination attributes, fields at T, historical ≤ T | FICO, current UPB, ever_delinquent |
| **Targets** | Future observations T+1 onward | next_3m_delinquency_flag |
| **Excluded** | Terminal indicators, future fields | zero_balance_removal_reason, upb_at_liquidation |

### 8.2 Per-Target Leakage Check

#### next_3m_delinquency_flag
- **Prediction time**: T
- **Target observation window**: T+1, T+2, T+3
- **Allowed features**: All fields at or before T
- **Forbidden in features**: delinquency_status at T+1/T+2/T+3, any zbr code, any future UPB
- **Verification**: Target computed via `groupby('loan_id').shift(-1/-2/-3)` on `is_delinquent`; feature columns contain no shifted future values

#### next_6m_delinquency_flag
- **Prediction time**: T
- **Target observation window**: T+1 through T+6
- **Same constraints as 3m target with extended window**

#### next_12m_default_flag
- **Prediction time**: T
- **Target observation window**: T+1 through T+12
- **Forbidden in features**: zero_balance_removal_reason (excluded from feature columns), any future zbr/termination fields
- **Verification**: zbr columns are in EXCLUDE_COLS and never appear in train/test feature files

#### next_12m_prepayment_flag
- **Same structure as default target**
- **Forbidden in features**: zero_balance_removal_reason (excluded)

#### next_state
- **Prediction time**: T
- **Target observation window**: T+1 only
- **Forbidden in features**: next month's state, zbr, future delinquency
- **Verification**: Computed via `groupby('loan_id').shift(-1)` on `current_state`

### 8.3 Rolling Feature Audit

| Feature | Calculation Method | Leakage Risk | Mitigation |
|---|---|---|---|
| `ever_delinquent` | `cummax()` within loan group | Would leak if computed on full history including future | Computed after sorting by period; `cummax()` only sees ≤ current row |
| `max_delinquency_to_date` | `cummax()` on numeric delinq | Same as above | Same mitigation |
| `delinquency_count_12m` | `rolling(12).sum()` | Window must not extend forward | Pandas `.rolling()` is backward-looking by default |
| `months_since_last_delinquency` | Cumulative group-counter | Must only count backward | Computed via `cumsum` + `cumcount` which are forward-only |
| `ever_modified` | `cummax()` on mod flag | Same as ever_delinquent | Same mitigation |

### 8.4 Train/Test Column Audit

| File | Contains Targets | Contains Future Fields | Status |
|---|---|---|---|
| `loan_monthly_performance_train.csv` | Yes (for training) | No | ✓ |
| `loan_monthly_performance_test.csv` | **No** | **No** | ✓ |
| `test_targets_holdout.csv` | Yes (evaluation only) | No | ✓ |

**Conclusion**: No future information leaks into the feature matrix.

---

## 9. Synthetic Anomaly Methodology

- **Source**: Real servicing observations from the training period
- **Baseline**: 10,000 randomly sampled real observations
- **Injected anomalies**: 200 total
  - BALANCE_DISCREPANCY: UPB perturbed ±5–15%
  - DELINQUENCY_STATUS_LAG: Delinquent status set to '00'
  - MISSING_MODIFICATION: Mod flag changed Y→N
- **Marking**: All synthetic records have `is_synthetic=True` and `anomaly_type`
- **Ground truth**: `servicer_updates_ground_truth.json` (evaluation only)

**These are intentionally created test cases, NOT real Fannie Mae anomalies.**

---

## 10. Validation Rules

16 rules defined in `validation_rules.json`. Categories:
- Impossible values (negative balance, out-of-range FICO/LTV/DTI)
- Required field violations
- Structural integrity (duplicate keys, invalid codes)
- Consistency checks (termination state vs. balance)
- Date relationship checks

---

## 11. Files Created

| File | Description | Rows |
|---|---|---|
| `data/processed/loan_monthly_performance_train.csv` | Train features + targets | 1,371,103 |
| `data/processed/loan_monthly_performance_test.csv` | Test features only | 285,776 |
| `data/processed/test_targets_holdout.csv` | Test ground truth (eval only) | 285,776 |
| `data/processed/loan_static_attributes.csv` | One row per loan | 25,000 |
| `data/processed/servicer_updates.csv` | Reconciliation data | ~10,200 |
| `data/processed/servicer_updates_ground_truth.json` | Anomaly labels | 200 |
| `data/processed/validation_rules.json` | Data quality rules | 16 |
| `data/processed/data_dictionary.md` | Complete documentation | — |
| `data/processed/macro_scenarios.csv` | Scenario assumptions | 18 |
| `data/processed/submission_template.csv` | Blank predictions | 285,776 |
| `reports/phase1_target_engineering_report.md` | This report | — |

---

## 12. Limitations

1. **XX (Unknown) status**: 21K+ observations have unknown delinquency. Target labels
   near these periods should be interpreted cautiously.
2. **Right-censoring**: Observations near the dataset boundary (late 2025) have incomplete
   forward windows for longer targets.
3. **Default definition**: Conservative — uses only unambiguous ZBR credit events.
   Some loans that are 90+ DPD but never formally terminated are NOT counted as defaults.
4. **Servicer anomalies**: Synthetic by design. Real anomaly patterns may differ.
5. **Macro scenarios**: Assumption structure only. No actual simulation in this phase.

---

**PHASE 1B COMPLETE**
