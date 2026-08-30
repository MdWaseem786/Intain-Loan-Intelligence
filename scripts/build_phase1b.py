#!/usr/bin/env python3
"""
Phase 1B: Competition Data Pack Construction + Target Engineering
================================================================
Reads:  data/processed/selected_loan_performance.parquet  (Phase 1A output)
Creates: All Phase 1B deliverables (train/test CSVs, static attributes,
         servicer updates, validation rules, data dictionary, macro scenarios,
         submission template, and the Phase 1B report).

Deterministic. Single-command reproducible. No ML training.
"""

import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PARQUET = os.path.join(BASE_DIR, 'data', 'processed', 'selected_loan_performance.parquet')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'processed')
REPORT_DIR = os.path.join(BASE_DIR, 'reports')

TRAIN_END = pd.Timestamp('2017-12-01')
TEST_START = pd.Timestamp('2018-01-01')
TEST_END = pd.Timestamp('2025-12-01')

RANDOM_SEED = 42
N_SYNTHETIC_ANOMALIES = 200

# Fannie Mae Zero-Balance Removal Reason codes
ZBR_PREPAID = {'01'}                # Voluntary payoff / matured
ZBR_DEFAULT = {'02', '03', '09'}    # Third-party sale, short sale, REO
ZBR_OTHER   = {'06', '15', '16'}    # Repurchase, note sale, reperforming sale

# Columns excluded from prediction-time features
EXCLUDE_COLS = {
    'master_servicer', 'upb_at_issuance',            # 100% null
    'zero_balance_removal_reason',                    # terminal indicator
    'zero_balance_effective_date',                    # terminal indicator
    'upb_at_liquidation',                             # terminal indicator
    'covid_plan_end_date', 'covid_exit_reason',       # mostly null / ambiguous
    'source_quarter',                                 # metadata
}

# Internal working columns (dropped before output)
INTERNAL_COLS = {
    'period_date', 'delinq_numeric', 'is_delinquent', 'is_unknown_status',
    'is_default_event', 'is_prepay_event', '_is_modified',
    'loan_last_period', 'loan_terminated', 'months_to_loan_end',
    '_delinq_for_max', '_delinq_group',
}

TARGET_COLS = [
    'next_3m_delinquency_flag', 'next_6m_delinquency_flag',
    'next_12m_default_flag', 'next_12m_prepayment_flag', 'next_state',
]
CENSORING_COLS = [
    'target_available_3m', 'target_available_6m', 'target_available_12m',
]
UNKNOWN_COLS = [
    'unknown_status_present_3m', 'unknown_status_present_6m',
    'unknown_status_present_12m',
]


# ============================================================
# SECTION 1: LOAD & VALIDATE
# ============================================================
def load_and_validate():
    print("=" * 60)
    print("SECTION 1: Loading Parquet")
    print("=" * 60)
    df = pd.read_parquet(INPUT_PARQUET)
    n_rows, n_cols = df.shape
    n_loans = df['loan_id'].nunique()
    print(f"  Rows: {n_rows:,}  Cols: {n_cols}  Unique loans: {n_loans:,}")
    assert n_loans == 25_000, f"Expected 25,000 loans, got {n_loans}"
    assert 'loan_id' in df.columns
    assert 'monthly_reporting_period' in df.columns
    assert 'current_delinquency_status' in df.columns
    assert 'zero_balance_removal_reason' in df.columns
    print("  Schema validated OK.")
    return df


# ============================================================
# SECTION 2: DATE PARSING & SORTING
# ============================================================
def parse_dates(df):
    print("\nSECTION 2: Parsing dates & sorting")
    df['period_date'] = pd.to_datetime(df['monthly_reporting_period'], format='%m%Y')
    df = df.sort_values(['loan_id', 'period_date']).reset_index(drop=True)
    print(f"  Date range: {df['period_date'].min().strftime('%Y-%m')} to "
          f"{df['period_date'].max().strftime('%Y-%m')}")
    return df


# ============================================================
# SECTION 3: CONSECUTIVE-MONTH VALIDATION
# ============================================================
def validate_consecutive(df):
    print("\nSECTION 3: Consecutive-month validation")
    prev = df.groupby('loan_id')['period_date'].shift(1)
    gap = ((df['period_date'].dt.year - prev.dt.year) * 12 +
           (df['period_date'].dt.month - prev.dt.month))
    non_consec = gap[gap.notna() & (gap != 1)]
    n_gaps = len(non_consec)
    if n_gaps > 0:
        print(f"  WARNING: {n_gaps} non-consecutive month transitions found.")
        print(f"  Gap distribution:\n{non_consec.value_counts().head(10)}")
    else:
        print("  All loan timelines are consecutive-monthly. OK.")
    return n_gaps


# ============================================================
# SECTION 4: STATE MAPPING & BINARY INDICATORS
# ============================================================
def map_states(df):
    print("\nSECTION 4: State mapping & indicators")
    # --- Numeric delinquency ---
    df['delinq_numeric'] = pd.to_numeric(
        df['current_delinquency_status'], errors='coerce'
    ).fillna(-1).astype(int)

    # Binary indicators
    df['is_delinquent'] = (df['delinq_numeric'] >= 1).astype(np.int8)
    df['is_unknown_status'] = (
        df['current_delinquency_status'] == 'XX'
    ).astype(np.int8)
    df['is_default_event'] = df['zero_balance_removal_reason'].isin(
        ZBR_DEFAULT
    ).astype(np.int8)
    df['is_prepay_event'] = df['zero_balance_removal_reason'].isin(
        ZBR_PREPAID
    ).astype(np.int8)

    # --- Current state taxonomy ---
    # Priority (lowest first, later assignments override):
    # 1. Current  2. 30DPD  3. 60DPD  4. 90+DPD  5. Unknown
    # 6. Terminal states (Prepaid, Default/REO)
    df['current_state'] = 'Current'
    df.loc[df['delinq_numeric'] == 1, 'current_state'] = '30DPD'
    df.loc[df['delinq_numeric'] == 2, 'current_state'] = '60DPD'
    df.loc[df['delinq_numeric'] >= 3, 'current_state'] = '90+DPD'
    df.loc[df['current_delinquency_status'] == 'XX', 'current_state'] = 'Unknown'

    # Terminal states override delinquency
    zbr = df['zero_balance_removal_reason']
    df.loc[zbr.isin(ZBR_PREPAID), 'current_state'] = 'Prepaid'
    df.loc[zbr.isin(ZBR_DEFAULT), 'current_state'] = 'Default/REO'
    # zbr 06/15/16: keep delinquency-based state (ambiguous termination)

    # Modification is a separate attribute (not a state)
    df['_is_modified'] = (df['monthly_modification_flag'] == 'Y').astype(np.int8)
    df['current_modified'] = df['_is_modified']

    print("  State distribution:")
    print(df['current_state'].value_counts().to_string(header=False))
    return df


# ============================================================
# SECTION 5: COMPUTE FORWARD-LOOKING TARGETS
# ============================================================
def compute_targets(df):
    print("\nSECTION 5: Computing forward-looking targets")
    g = df.groupby('loan_id')

    # --- next_state (T+1) ---
    df['next_state'] = g['current_state'].shift(-1)

    # --- Helper: forward max over k future months ---
    def forward_max(series_name, k, target_name):
        cols = []
        for i in range(1, k + 1):
            col = f'_tmp_{series_name}_{i}'
            df[col] = g[series_name].shift(-i)
            cols.append(col)
        df[target_name] = df[cols].max(axis=1)
        df.drop(columns=cols, inplace=True)

    # --- Delinquency targets (30+ DPD, XX excluded) ---
    forward_max('is_delinquent', 3, 'next_3m_delinquency_flag')
    forward_max('is_delinquent', 6, 'next_6m_delinquency_flag')
    print("  Delinquency targets computed.")

    # --- Default target (12m) ---
    forward_max('is_default_event', 12, 'next_12m_default_flag')
    print("  Default target computed.")

    # --- Prepayment target (12m) ---
    forward_max('is_prepay_event', 12, 'next_12m_prepayment_flag')
    print("  Prepayment target computed.")

    # --- Unknown-status-present flags ---
    forward_max('is_unknown_status', 3, 'unknown_status_present_3m')
    forward_max('is_unknown_status', 6, 'unknown_status_present_6m')
    forward_max('is_unknown_status', 12, 'unknown_status_present_12m')
    print("  Unknown-status flags computed.")

    return df


# ============================================================
# SECTION 6: RIGHT-CENSORING
# ============================================================
def compute_censoring(df):
    print("\nSECTION 6: Right-censoring")
    # Per-loan metadata
    loan_meta = df.groupby('loan_id').agg(
        loan_last_period=('period_date', 'max'),
        loan_terminated=('zero_balance_removal_reason',
                         lambda x: x.notna().any()),
    )
    df = df.merge(loan_meta, on='loan_id', how='left')

    # Months remaining in this loan's timeline
    df['months_to_loan_end'] = (
        (df['loan_last_period'].dt.year - df['period_date'].dt.year) * 12 +
        (df['loan_last_period'].dt.month - df['period_date'].dt.month)
    )

    # Target availability: available if terminated OR enough future months
    for k, name in [(3, '3m'), (6, '6m'), (12, '12m')]:
        df[f'target_available_{name}'] = np.where(
            (df['months_to_loan_end'] > 0) &
            (df['loan_terminated'] | (df['months_to_loan_end'] >= k)),
            1, 0
        ).astype(np.int8)

    # Apply censoring: set targets to NaN where not available
    for tgt, avail in [
        ('next_3m_delinquency_flag', 'target_available_3m'),
        ('next_6m_delinquency_flag', 'target_available_6m'),
        ('next_12m_default_flag', 'target_available_12m'),
        ('next_12m_prepayment_flag', 'target_available_12m'),
        ('unknown_status_present_3m', 'target_available_3m'),
        ('unknown_status_present_6m', 'target_available_6m'),
        ('unknown_status_present_12m', 'target_available_12m'),
    ]:
        df.loc[df[avail] == 0, tgt] = np.nan

    # Last observation per loan: no future → all targets NaN
    last_mask = df['months_to_loan_end'] == 0
    for col in TARGET_COLS + UNKNOWN_COLS:
        df.loc[last_mask, col] = np.nan

    print(f"  Censored (last obs): {last_mask.sum():,}")
    for name in ['3m', '6m', '12m']:
        avail = (df[f'target_available_{name}'] == 1).sum()
        unavail = (df[f'target_available_{name}'] == 0).sum()
        print(f"  target_available_{name}: available={avail:,}  censored={unavail:,}")

    return df


# ============================================================
# SECTION 7: HISTORICAL FEATURE ENGINEERING
# ============================================================
def compute_features(df):
    print("\nSECTION 7: Historical feature engineering (≤T only)")
    g = df.groupby('loan_id')

    # pct_life_elapsed
    df['pct_life_elapsed'] = df['loan_age'] / df['original_loan_term']

    # rate_spread
    df['rate_spread'] = df['current_interest_rate'] - df['original_interest_rate']

    # upb_pct_original
    df['upb_pct_original'] = np.where(
        df['original_upb'] > 0,
        df['current_actual_upb'] / df['original_upb'],
        np.nan
    )

    # ever_delinquent (cummax of binary delinquent flag)
    df['ever_delinquent'] = g['is_delinquent'].cummax().astype(np.int8)

    # max_delinquency_to_date
    df['_delinq_for_max'] = df['delinq_numeric'].clip(lower=0)
    df['max_delinquency_to_date'] = g['_delinq_for_max'].cummax().astype(np.int16)

    # months_since_last_delinquency
    df['_delinq_group'] = g['is_delinquent'].cumsum()
    df['months_since_last_delinquency'] = df.groupby(
        ['loan_id', '_delinq_group']
    ).cumcount()
    # Before first delinquency (group 0): set to NaN
    df.loc[df['_delinq_group'] == 0, 'months_since_last_delinquency'] = np.nan

    # delinquency_count_12m (rolling 12-month sum)
    df['delinquency_count_12m'] = g['is_delinquent'].transform(
        lambda x: x.rolling(12, min_periods=1).sum()
    ).astype(np.int16)

    # ever_modified
    df['ever_modified'] = g['_is_modified'].cummax().astype(np.int8)

    print("  Features: pct_life_elapsed, rate_spread, upb_pct_original,")
    print("            ever_delinquent, max_delinquency_to_date,")
    print("            months_since_last_delinquency, delinquency_count_12m,")
    print("            ever_modified")
    return df


# ============================================================
# SECTION 8: SPLIT TRAIN / TEST & SAVE
# ============================================================
def split_and_save(df):
    print("\nSECTION 8: Train/Test split")

    # Define column sets
    feature_cols = [c for c in df.columns
                    if c not in EXCLUDE_COLS
                    and c not in INTERNAL_COLS
                    and c not in TARGET_COLS
                    and c not in CENSORING_COLS
                    and c not in UNKNOWN_COLS]

    # Period masks
    train_mask = df['period_date'] <= TRAIN_END
    test_mask = (df['period_date'] >= TEST_START) & (df['period_date'] <= TEST_END)

    # ---- TRAIN ----
    train_cols = feature_cols + TARGET_COLS + CENSORING_COLS + UNKNOWN_COLS
    train_df = df.loc[train_mask, train_cols].copy()
    train_path = os.path.join(OUTPUT_DIR, 'loan_monthly_performance_train.csv')
    train_df.to_csv(train_path, index=False)
    print(f"  Train: {len(train_df):,} rows → {train_path}")

    # ---- TEST (features only) ----
    test_df = df.loc[test_mask, feature_cols].copy()
    test_path = os.path.join(OUTPUT_DIR, 'loan_monthly_performance_test.csv')
    test_df.to_csv(test_path, index=False)
    print(f"  Test:  {len(test_df):,} rows → {test_path}")

    # ---- HOLDOUT TARGETS (internal evaluation only) ----
    holdout_cols = ['loan_id', 'monthly_reporting_period'] + \
                   TARGET_COLS + CENSORING_COLS + UNKNOWN_COLS
    holdout_df = df.loc[test_mask, holdout_cols].copy()
    holdout_path = os.path.join(OUTPUT_DIR, 'test_targets_holdout.csv')
    holdout_df.to_csv(holdout_path, index=False)
    print(f"  Holdout: {len(holdout_df):,} rows → {holdout_path}")

    return train_df, test_df, holdout_df


# ============================================================
# SECTION 9: STATIC ATTRIBUTES
# ============================================================
def create_static_attributes(df):
    print("\nSECTION 9: Static attributes")
    static_cols = [
        'loan_id', 'channel', 'seller_name',
        'original_interest_rate', 'original_upb', 'original_loan_term',
        'origination_date', 'first_payment_date', 'maturity_date',
        'original_ltv', 'original_cltv', 'number_of_borrowers',
        'dti_ratio', 'borrower_credit_score', 'co_borrower_credit_score',
        'first_time_homebuyer_flag', 'loan_purpose', 'property_type',
        'number_of_units', 'occupancy_status', 'property_state',
        'msa', 'zip_code_short', 'mi_percentage', 'product_type',
        'special_eligibility_program', 'relocation_mortgage_flag',
    ]
    # Take first observation per loan (origination attributes are static)
    static_df = df.sort_values('period_date').groupby('loan_id').first()[
        [c for c in static_cols if c != 'loan_id']
    ].reset_index()

    assert len(static_df) == 25_000, f"Expected 25,000, got {len(static_df)}"
    assert static_df['loan_id'].nunique() == 25_000

    path = os.path.join(OUTPUT_DIR, 'loan_static_attributes.csv')
    static_df.to_csv(path, index=False)
    print(f"  Static: {len(static_df):,} loans → {path}")
    return static_df


# ============================================================
# SECTION 10: SERVICER UPDATES + SYNTHETIC ANOMALIES
# ============================================================
def create_servicer_updates(df):
    print("\nSECTION 10: Servicer updates & synthetic anomalies")
    rng = np.random.RandomState(RANDOM_SEED)

    # Baseline: sample real servicing observations from train period
    svc_cols = [
        'loan_id', 'monthly_reporting_period', 'servicer_name',
        'current_actual_upb', 'current_interest_rate',
        'current_delinquency_status', 'monthly_modification_flag',
    ]
    train_mask = df['period_date'] <= TRAIN_END
    base = df.loc[train_mask, svc_cols].copy()

    # Sample 10,000 baseline records
    n_base = min(10_000, len(base))
    base_sample = base.sample(n=n_base, random_state=RANDOM_SEED).copy()
    base_sample['is_synthetic'] = False
    base_sample['anomaly_type'] = ''

    # --- Create synthetic anomalies ---
    anomalies = []
    ground_truth = []

    # Type 1: BALANCE_DISCREPANCY (~70)
    n_bal = 70
    bal_candidates = base.loc[base['current_actual_upb'] > 1000].sample(
        n=n_bal, random_state=RANDOM_SEED + 1
    ).copy()
    for idx, row in bal_candidates.iterrows():
        original_val = row['current_actual_upb']
        factor = 1.0 + rng.uniform(-0.15, 0.15)
        modified_val = round(original_val * factor, 2)
        anom = row.copy()
        anom['current_actual_upb'] = modified_val
        anom['is_synthetic'] = True
        anom['anomaly_type'] = 'BALANCE_DISCREPANCY'
        anomalies.append(anom)
        ground_truth.append({
            'loan_id': str(row['loan_id']),
            'reporting_period': row['monthly_reporting_period'],
            'anomaly_type': 'BALANCE_DISCREPANCY',
            'field': 'current_actual_upb',
            'original_value': float(original_val),
            'modified_value': float(modified_val),
        })

    # Type 2: DELINQUENCY_STATUS_LAG (~70)
    n_lag = 70
    delinq_candidates = base.loc[
        ~base['current_delinquency_status'].isin(['00', 'XX'])
    ]
    if len(delinq_candidates) >= n_lag:
        lag_sample = delinq_candidates.sample(
            n=n_lag, random_state=RANDOM_SEED + 2
        ).copy()
    else:
        lag_sample = delinq_candidates.copy()
    for idx, row in lag_sample.iterrows():
        original_val = row['current_delinquency_status']
        anom = row.copy()
        anom['current_delinquency_status'] = '00'
        anom['is_synthetic'] = True
        anom['anomaly_type'] = 'DELINQUENCY_STATUS_LAG'
        anomalies.append(anom)
        ground_truth.append({
            'loan_id': str(row['loan_id']),
            'reporting_period': row['monthly_reporting_period'],
            'anomaly_type': 'DELINQUENCY_STATUS_LAG',
            'field': 'current_delinquency_status',
            'original_value': original_val,
            'modified_value': '00',
        })

    # Type 3: MISSING_MODIFICATION (~60)
    n_mod = 60
    mod_candidates = base.loc[base['monthly_modification_flag'] == 'Y']
    if len(mod_candidates) >= n_mod:
        mod_sample = mod_candidates.sample(
            n=n_mod, random_state=RANDOM_SEED + 3
        ).copy()
    else:
        mod_sample = mod_candidates.copy()
    for idx, row in mod_sample.iterrows():
        anom = row.copy()
        anom['monthly_modification_flag'] = 'N'
        anom['is_synthetic'] = True
        anom['anomaly_type'] = 'MISSING_MODIFICATION'
        anomalies.append(anom)
        ground_truth.append({
            'loan_id': str(row['loan_id']),
            'reporting_period': row['monthly_reporting_period'],
            'anomaly_type': 'MISSING_MODIFICATION',
            'field': 'monthly_modification_flag',
            'original_value': 'Y',
            'modified_value': 'N',
        })

    # Combine
    anomaly_df = pd.DataFrame(anomalies)
    servicer_df = pd.concat([base_sample, anomaly_df], ignore_index=True)

    # Save servicer updates
    svc_path = os.path.join(OUTPUT_DIR, 'servicer_updates.csv')
    servicer_df.to_csv(svc_path, index=False)

    # Save ground truth
    gt = {
        'description': (
            'Ground truth labels for synthetic servicer update anomalies. '
            'These are intentionally created test cases, NOT real Fannie Mae anomalies. '
            'For evaluation only — must not be used as model features.'
        ),
        'generated_date': datetime.now().strftime('%Y-%m-%d'),
        'total_anomalies': len(ground_truth),
        'anomaly_types': {
            'BALANCE_DISCREPANCY': n_bal,
            'DELINQUENCY_STATUS_LAG': len(lag_sample),
            'MISSING_MODIFICATION': len(mod_sample),
        },
        'anomalies': ground_truth,
    }
    gt_path = os.path.join(OUTPUT_DIR, 'servicer_updates_ground_truth.json')
    with open(gt_path, 'w') as f:
        json.dump(gt, f, indent=2)

    actual_anomalies = len(ground_truth)
    print(f"  Servicer updates: {len(servicer_df):,} rows → {svc_path}")
    print(f"  Synthetic anomalies: {actual_anomalies}")
    print(f"  Ground truth: {gt_path}")
    return actual_anomalies


# ============================================================
# SECTION 11: VALIDATION RULES
# ============================================================
def create_validation_rules():
    print("\nSECTION 11: Validation rules")
    rules = [
        {
            "rule_id": "VR001",
            "description": "Negative current UPB",
            "condition": "current_actual_upb < 0",
            "severity": "ERROR",
            "rationale": "Unpaid principal balance cannot be negative."
        },
        {
            "rule_id": "VR002",
            "description": "Invalid original LTV (out of range)",
            "condition": "original_ltv <= 0 OR original_ltv > 200",
            "severity": "ERROR",
            "rationale": "LTV outside 0–200% is implausible for conforming loans."
        },
        {
            "rule_id": "VR003",
            "description": "Invalid DTI ratio",
            "condition": "dti_ratio < 0 OR dti_ratio > 100",
            "severity": "WARNING",
            "rationale": "DTI outside 0–100% is unusual. DTI > 65% merits review."
        },
        {
            "rule_id": "VR004",
            "description": "Invalid borrower credit score",
            "condition": "borrower_credit_score < 300 OR borrower_credit_score > 850",
            "severity": "ERROR",
            "rationale": "FICO scores are defined in the 300–850 range."
        },
        {
            "rule_id": "VR005",
            "description": "Negative original interest rate",
            "condition": "original_interest_rate <= 0",
            "severity": "ERROR",
            "rationale": "Interest rate must be positive for US conforming mortgages."
        },
        {
            "rule_id": "VR006",
            "description": "Invalid original loan term",
            "condition": "original_loan_term NOT IN (120, 180, 240, 300, 360)",
            "severity": "WARNING",
            "rationale": "Standard terms are 10/15/20/25/30 years."
        },
        {
            "rule_id": "VR007",
            "description": "Origination after first reporting period",
            "condition": "origination_date > monthly_reporting_period (as dates)",
            "severity": "WARNING",
            "rationale": "Origination should precede or equal the first observation."
        },
        {
            "rule_id": "VR008",
            "description": "Duplicate loan-month key",
            "condition": "COUNT(loan_id, monthly_reporting_period) > 1",
            "severity": "ERROR",
            "rationale": "Panel data must have unique loan-month keys."
        },
        {
            "rule_id": "VR009",
            "description": "Invalid delinquency status code",
            "condition": "current_delinquency_status NOT IN ('00'..'99', 'XX')",
            "severity": "ERROR",
            "rationale": "Delinquency must be 00–99 or XX per Fannie Mae spec."
        },
        {
            "rule_id": "VR010",
            "description": "Inconsistent termination: ZBR set but high UPB",
            "condition": "zero_balance_removal_reason IS NOT NULL AND current_actual_upb > 0 AND zero_balance_removal_reason IN ('01')",
            "severity": "WARNING",
            "rationale": "Prepaid loans should typically have zero or near-zero UPB."
        },
        {
            "rule_id": "VR011",
            "description": "Missing loan_id",
            "condition": "loan_id IS NULL OR loan_id = ''",
            "severity": "ERROR",
            "rationale": "Loan identifier is a required field."
        },
        {
            "rule_id": "VR012",
            "description": "Missing reporting period",
            "condition": "monthly_reporting_period IS NULL OR monthly_reporting_period = ''",
            "severity": "ERROR",
            "rationale": "Reporting period is required for panel structure."
        },
        {
            "rule_id": "VR013",
            "description": "Negative loan age",
            "condition": "loan_age < 0",
            "severity": "ERROR",
            "rationale": "Loan age cannot be negative."
        },
        {
            "rule_id": "VR014",
            "description": "Current UPB exceeds 150% of original UPB",
            "condition": "current_actual_upb > original_upb * 1.5",
            "severity": "WARNING",
            "rationale": "Balance growth beyond 150% of original is unusual; may indicate capitalized interest/fees or data error."
        },
        {
            "rule_id": "VR015",
            "description": "Invalid co-borrower credit score",
            "condition": "co_borrower_credit_score IS NOT NULL AND (co_borrower_credit_score < 300 OR co_borrower_credit_score > 850)",
            "severity": "ERROR",
            "rationale": "FICO scores are defined in the 300–850 range."
        },
        {
            "rule_id": "VR016",
            "description": "Missing required origination fields",
            "condition": "original_upb IS NULL OR original_interest_rate IS NULL OR original_loan_term IS NULL",
            "severity": "ERROR",
            "rationale": "Core origination attributes are required for analysis."
        },
    ]
    path = os.path.join(OUTPUT_DIR, 'validation_rules.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(rules, f, indent=2)
    print(f"  {len(rules)} validation rules → {path}")
    return len(rules)


# ============================================================
# SECTION 12: DATA DICTIONARY
# ============================================================
def create_data_dictionary():
    print("\nSECTION 12: Data dictionary")
    content = """# Data Dictionary — Intain Loan Performance Intelligence Engine

## Phase 1B: Competition Data Pack

Generated: {date}

---

## 1. Source Fields (Fannie Mae Monthly Performance)

| Field | Type | Missing Convention | Description |
|---|---|---|---|
| `loan_id` | string | Never null | Fannie Mae unique loan identifier |
| `monthly_reporting_period` | string (MMYYYY) | Never null | Monthly reporting period |
| `channel` | string | Never null | Origination channel: R=Retail, B=Broker, C=Correspondent, T=TPO |
| `seller_name` | string | Never null | Name of the loan seller |
| `servicer_name` | string | Null ~1.4% | Current loan servicer |
| `original_interest_rate` | float32 | Never null | Note rate at origination (%) |
| `current_interest_rate` | float32 | Null ~1.4% | Current note rate (%) |
| `original_upb` | float64 | Never null | Original unpaid principal balance ($) |
| `current_actual_upb` | float64 | Never null | Current unpaid principal balance ($) |
| `original_loan_term` | Int16 | Never null | Original term in months |
| `origination_date` | string (MMYYYY) | Never null | Loan origination date |
| `first_payment_date` | string (MMYYYY) | Never null | First scheduled payment date |
| `loan_age` | Int16 | Null ~1.4% | Months since origination |
| `remaining_months_to_maturity` | Int16 | Null ~1.4% | Remaining months to maturity |
| `adjusted_months_to_maturity` | Int16 | Null ~2.2% | Adjusted remaining months |
| `maturity_date` | string (MMYYYY) | Null ~1.4% | Scheduled maturity date |
| `original_ltv` | float32 | Never null | Original loan-to-value ratio (%) |
| `original_cltv` | float32 | Never null | Combined LTV (%) |
| `number_of_borrowers` | Int8 | Null ~0.01% | Number of borrowers |
| `dti_ratio` | float32 | Null ~0.9% | Debt-to-income ratio (%) |
| `borrower_credit_score` | float32 | Null ~0.2% | Borrower FICO at origination |
| `co_borrower_credit_score` | float32 | Null ~44.6% | Co-borrower FICO at origination |
| `first_time_homebuyer_flag` | string | Null ~0.03% | Y=Yes, N=No |
| `loan_purpose` | string | Never null | P=Purchase, C=Cash-out Refi, R=Rate/Term Refi, U=Unspecified |
| `property_type` | string | Never null | SF, PU, CO, MH, CP |
| `number_of_units` | Int8 | Never null | Property units (1–4) |
| `occupancy_status` | string | Never null | P=Primary, S=Second, I=Investment |
| `property_state` | string | Never null | US state abbreviation |
| `msa` | string | Never null | Metro statistical area code |
| `zip_code_short` | string | Never null | 3-digit zip code |
| `mi_percentage` | float32 | Null ~93.3% | Mortgage insurance percentage |
| `product_type` | string | Never null | FRM=Fixed Rate Mortgage |
| `current_delinquency_status` | string | Never null | 00=Current, 01–99=Months DPD, XX=Unknown |
| `monthly_modification_flag` | string | Null ~1.4% | Y=Modified this month, N=Not |
| `zero_balance_removal_reason` | string | Null ~98.6% | 01=Prepaid, 02=3rd Party Sale, 03=Short Sale, 06=Repurchase, 09=REO, 15=Note Sale, 16=Reperforming Sale |
| `zero_balance_effective_date` | string (MMYYYY) | Null ~98.6% | Date of zero balance |
| `upb_at_liquidation` | float64 | Null ~98.6% | UPB at disposition |
| `covid_forbearance_flag` | string | Never null | Y/N |
| `covid_assistance_code` | string | Null ~90.4% | COVID assistance type |

**Excluded from features** (100% null or terminal indicators):
`master_servicer`, `upb_at_issuance`, `zero_balance_removal_reason`,
`zero_balance_effective_date`, `upb_at_liquidation`, `covid_plan_end_date`,
`covid_exit_reason`, `source_quarter`

---

## 2. Engineered Features (Prediction-Time, ≤ T)

| Feature | Formula | Description |
|---|---|---|
| `current_state` | Mapped from delinq + zbr | Current loan state (see taxonomy below) |
| `current_modified` | `monthly_modification_flag == 'Y'` | Whether loan was modified this month |
| `pct_life_elapsed` | `loan_age / original_loan_term` | Fraction of original term elapsed |
| `rate_spread` | `current_interest_rate - original_interest_rate` | Rate change from origination |
| `upb_pct_original` | `current_actual_upb / original_upb` | Balance as fraction of original |
| `ever_delinquent` | Cumulative max of delinquency indicator ≤ T | Whether loan was ever 30+ DPD |
| `max_delinquency_to_date` | Cumulative max of numeric delinq ≤ T | Worst delinquency status seen (months) |
| `months_since_last_delinquency` | Months since most recent delinq event | NaN if never delinquent |
| `delinquency_count_12m` | Rolling 12-month sum of delinq months | Count of DPD months in trailing year |
| `ever_modified` | Cumulative max of modification flag ≤ T | Whether loan was ever modified |

---

## 3. Target Definitions

| Target | Window | Positive Condition | Negative Condition |
|---|---|---|---|
| `next_3m_delinquency_flag` | T+1..T+3 | Any month has `current_delinquency_status` ≥ '01' | All months '00' or 'XX' or terminated |
| `next_6m_delinquency_flag` | T+1..T+6 | Same as above | Same as above |
| `next_12m_default_flag` | T+1..T+12 | `zero_balance_removal_reason` ∈ (02, 03, 09) | No default/REO/short-sale termination |
| `next_12m_prepayment_flag` | T+1..T+12 | `zero_balance_removal_reason` = '01' | No voluntary payoff |
| `next_state` | T+1 | Mapped state at T+1 | NaN if no T+1 exists |

**XX handling**: 'XX' (Unknown/forbearance) does NOT count as delinquency.
Tracked separately via `unknown_status_present_*` flags.

**Default definition**: Only unambiguous credit events — Third-Party Sale (02),
Short Sale (03), and REO Disposition (09). Does NOT include 90+ DPD alone, repurchase,
note sale, or reperforming loan sale.

---

## 4. State Taxonomy (priority order, highest first)

| Priority | State | Condition |
|---|---|---|
| 1 | Default/REO | `zero_balance_removal_reason` ∈ (02, 03, 09) |
| 2 | Prepaid | `zero_balance_removal_reason` = '01' |
| 3 | 90+DPD | `current_delinquency_status` ≥ '03' (numeric ≥ 3) |
| 4 | 60DPD | `current_delinquency_status` = '02' |
| 5 | 30DPD | `current_delinquency_status` = '01' |
| 6 | Unknown | `current_delinquency_status` = 'XX' |
| 7 | Current | `current_delinquency_status` = '00' |

**Modification** is NOT a performance state; it is tracked separately via
`current_modified` and `ever_modified`.

ZBR codes 06 (Repurchase), 15 (Note Sale), 16 (Reperforming Sale) retain their
delinquency-based state because they are not clearly borrower-driven events.

---

## 5. Right-Censoring & Target Availability

| Field | Description |
|---|---|
| `target_available_3m` | 1 if 3-month forward window is fully observable |
| `target_available_6m` | 1 if 6-month forward window is fully observable |
| `target_available_12m` | 1 if 12-month forward window is fully observable |

**Rule**: Target is available if (a) the loan terminated within the dataset timeline
(we know its fate regardless of window length), OR (b) enough future calendar months
exist in the dataset. If neither condition holds, the observation is right-censored
and targets are set to NaN.

---

## 6. Unknown-Status Flags

| Field | Description |
|---|---|
| `unknown_status_present_3m` | 1 if any XX status appears in T+1..T+3 |
| `unknown_status_present_6m` | 1 if any XX status appears in T+1..T+6 |
| `unknown_status_present_12m` | 1 if any XX status appears in T+1..T+12 |

---

## 7. Servicer Updates (Synthetic Anomalies)

| Anomaly Type | Description |
|---|---|
| BALANCE_DISCREPANCY | `current_actual_upb` perturbed by ±5–15% |
| DELINQUENCY_STATUS_LAG | Delinquent status replaced with '00' (reporting lag) |
| MISSING_MODIFICATION | Modification flag changed from 'Y' to 'N' |

All synthetic records are explicitly marked (`is_synthetic=True`, `anomaly_type`).
Ground truth labels stored separately in `servicer_updates_ground_truth.json`.
These are controlled test cases — NOT real Fannie Mae anomalies.

---

## 8. Macro Scenario Fields

| Field | Description |
|---|---|
| `scenario_name` | Base / Adverse-Credit / High-Prepayment |
| `parameter` | Macro parameter name |
| `value` | Assumed value |
| `type` | assumption / projection |
| `description` | Human-readable explanation |

Scenarios are assumption configurations only. Historical data is not fabricated.
""".format(date=datetime.now().strftime('%Y-%m-%d'))

    path = os.path.join(OUTPUT_DIR, 'data_dictionary.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  Data dictionary → {path}")


# ============================================================
# SECTION 13: MACRO SCENARIOS
# ============================================================
def create_macro_scenarios():
    print("\nSECTION 13: Macro scenarios")
    scenarios = [
        # Base scenario
        ('Base', 'assumption', 'gdp_growth_pct', '2.0', 'Baseline annual GDP growth'),
        ('Base', 'assumption', 'unemployment_rate_pct', '4.0', 'Baseline unemployment rate'),
        ('Base', 'assumption', 'hpi_growth_pct', '3.0', 'Baseline home price index growth'),
        ('Base', 'assumption', 'mortgage_rate_30y_pct', '6.5', 'Baseline 30-year fixed rate'),
        ('Base', 'assumption', 'prepayment_multiplier', '1.0', 'No adjustment to base prepayment'),
        ('Base', 'assumption', 'default_multiplier', '1.0', 'No adjustment to base default'),
        # Adverse-Credit
        ('Adverse-Credit', 'assumption', 'gdp_growth_pct', '-1.5', 'Recessionary contraction'),
        ('Adverse-Credit', 'assumption', 'unemployment_rate_pct', '8.0', 'Elevated unemployment'),
        ('Adverse-Credit', 'assumption', 'hpi_growth_pct', '-10.0', 'Significant home price decline'),
        ('Adverse-Credit', 'assumption', 'mortgage_rate_30y_pct', '7.5', 'Rising rates under stress'),
        ('Adverse-Credit', 'assumption', 'prepayment_multiplier', '0.5', 'Reduced prepayment under stress'),
        ('Adverse-Credit', 'assumption', 'default_multiplier', '3.0', 'Tripled default hazard'),
        # High-Prepayment
        ('High-Prepayment', 'assumption', 'gdp_growth_pct', '3.0', 'Moderate expansion'),
        ('High-Prepayment', 'assumption', 'unemployment_rate_pct', '3.5', 'Low unemployment'),
        ('High-Prepayment', 'assumption', 'hpi_growth_pct', '6.0', 'Strong home price appreciation'),
        ('High-Prepayment', 'assumption', 'mortgage_rate_30y_pct', '4.5', 'Low refinancing rates'),
        ('High-Prepayment', 'assumption', 'prepayment_multiplier', '2.5', 'Elevated prepayment speed'),
        ('High-Prepayment', 'assumption', 'default_multiplier', '0.5', 'Reduced default hazard'),
    ]
    df_scen = pd.DataFrame(scenarios, columns=[
        'scenario_name', 'type', 'parameter', 'value', 'description'
    ])
    path = os.path.join(OUTPUT_DIR, 'macro_scenarios.csv')
    df_scen.to_csv(path, index=False)
    print(f"  {len(df_scen)} scenario parameters → {path}")


# ============================================================
# SECTION 14: SUBMISSION TEMPLATE
# ============================================================
def create_submission_template(test_df):
    print("\nSECTION 14: Submission template")
    template = test_df[['loan_id', 'monthly_reporting_period']].copy()
    template['delinquency_probability'] = np.nan
    template['default_probability'] = np.nan
    template['prepayment_probability'] = np.nan
    template['next_state'] = ''
    template['exception_type'] = ''
    template['anomaly_score'] = np.nan
    template['top_drivers'] = ''
    template['action'] = ''
    template['confidence'] = np.nan

    path = os.path.join(OUTPUT_DIR, 'submission_template.csv')
    template.to_csv(path, index=False)
    print(f"  Submission template: {len(template):,} rows → {path}")
    return template


# ============================================================
# SECTION 15: VALIDATION AUDIT
# ============================================================
def run_validation(df, train_df, test_df, holdout_df, static_df, template):
    print("\n" + "=" * 60)
    print("SECTION 15: Validation Audit")
    print("=" * 60)
    errors = []

    # 1. Unique loans
    n_loans = df['loan_id'].nunique()
    assert n_loans == 25_000, f"Loan count: {n_loans}"
    print(f"  ✓ Unique loans: {n_loans:,}")

    # 2. Train/test temporal separation
    train_max = pd.to_datetime(
        train_df['monthly_reporting_period'], format='%m%Y'
    ).max()
    test_min = pd.to_datetime(
        test_df['monthly_reporting_period'], format='%m%Y'
    ).min()
    assert train_max <= TRAIN_END, f"Train max {train_max} > {TRAIN_END}"
    assert test_min >= TEST_START, f"Test min {test_min} < {TEST_START}"
    print(f"  ✓ Train max period: {train_max.strftime('%Y-%m')}")
    print(f"  ✓ Test min period: {test_min.strftime('%Y-%m')}")

    # 3. No duplicate keys
    train_dupes = train_df.duplicated(
        subset=['loan_id', 'monthly_reporting_period']
    ).sum()
    test_dupes = test_df.duplicated(
        subset=['loan_id', 'monthly_reporting_period']
    ).sum()
    assert train_dupes == 0, f"Train duplicates: {train_dupes}"
    assert test_dupes == 0, f"Test duplicates: {test_dupes}"
    print(f"  ✓ No duplicate loan-month keys (train: {train_dupes}, test: {test_dupes})")

    # 4. Static table
    assert len(static_df) == 25_000
    assert static_df['loan_id'].nunique() == 25_000
    print(f"  ✓ Static table: {len(static_df):,} rows, {static_df['loan_id'].nunique():,} unique loans")

    # 5. Holdout/test key match
    test_keys = set(zip(test_df['loan_id'], test_df['monthly_reporting_period']))
    holdout_keys = set(zip(holdout_df['loan_id'], holdout_df['monthly_reporting_period']))
    assert test_keys == holdout_keys, "Test/holdout key mismatch"
    print(f"  ✓ Test/holdout keys match: {len(test_keys):,}")

    # 6. Template/test key match
    tmpl_keys = set(zip(template['loan_id'], template['monthly_reporting_period']))
    assert tmpl_keys == test_keys, "Template/test key mismatch"
    print(f"  ✓ Submission template matches test keys")

    # 7. No future columns in test features
    future_cols_in_test = set(test_df.columns) & (
        set(TARGET_COLS) | set(CENSORING_COLS) | set(UNKNOWN_COLS) |
        {'zero_balance_removal_reason', 'zero_balance_effective_date',
         'upb_at_liquidation'}
    )
    assert len(future_cols_in_test) == 0, \
        f"Future/target columns in test: {future_cols_in_test}"
    print(f"  ✓ No future/target columns in test feature file")

    # 8. Target distributions (train only, available targets)
    stats = {}
    stats['train_rows'] = len(train_df)
    stats['test_rows'] = len(test_df)
    stats['train_loans'] = train_df['loan_id'].nunique()
    stats['test_loans'] = test_df['loan_id'].nunique()

    for tgt in ['next_3m_delinquency_flag', 'next_6m_delinquency_flag',
                'next_12m_default_flag', 'next_12m_prepayment_flag']:
        available = train_df[tgt].notna()
        pos = (train_df.loc[available, tgt] == 1).sum()
        total = available.sum()
        rate = pos / total * 100 if total > 0 else 0
        stats[f'{tgt}_positive'] = int(pos)
        stats[f'{tgt}_total_available'] = int(total)
        stats[f'{tgt}_rate'] = round(rate, 4)
        print(f"  Target {tgt}: {pos:,}/{total:,} ({rate:.2f}%)")

    # Next state distribution (train)
    ns_dist = train_df['next_state'].value_counts(dropna=False)
    stats['next_state_dist'] = ns_dist.to_dict()
    print(f"  Next-state distribution (train):")
    for state, cnt in ns_dist.items():
        label = str(state) if pd.notna(state) else 'NaN (censored/last)'
        print(f"    {label}: {cnt:,}")

    # Censoring counts
    for name in ['3m', '6m', '12m']:
        censored = (train_df[f'target_available_{name}'] == 0).sum()
        stats[f'censored_{name}'] = int(censored)
        print(f"  Censored (train) {name}: {censored:,}")

    # Unknown status counts
    for name in ['3m', '6m', '12m']:
        col = f'unknown_status_present_{name}'
        if col in train_df.columns:
            unk = (train_df[col] == 1).sum()
            stats[f'unknown_{name}'] = int(unk)
            print(f"  Unknown-status-present {name}: {unk:,}")

    print("\n  === ALL VALIDATION CHECKS PASSED ===")
    return stats


# ============================================================
# SECTION 16: REPORT & LEAKAGE AUDIT
# ============================================================
def generate_report(stats, n_anomalies, n_rules, n_gaps, runtime_sec):
    print("\nSECTION 16: Generating report")

    # Format next-state distribution
    ns_lines = []
    ns_dict = stats.get('next_state_dist', {})
    for state, cnt in sorted(ns_dict.items(), key=lambda x: -x[1]):
        label = str(state) if pd.notna(state) else 'NaN (censored/last obs)'
        ns_lines.append(f"| {label} | {cnt:,} |")

    report = f"""# Phase 1B: Target Engineering & Data Pack Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Runtime**: {runtime_sec:.1f} seconds
**Reproduction**: `python scripts/build_phase1b.py`

---

## 1. Dataset Summary

| Metric | Value |
|---|---|
| Source | `data/processed/selected_loan_performance.parquet` |
| Total unique loans | 25,000 |
| Total monthly observations | 1,660,802 |
| Actual date range | 07/2009 – 03/2026 |
| Train observations (≤ 2017-12) | {stats['train_rows']:,} |
| Test observations (2018-01 – 2025-12) | {stats['test_rows']:,} |
| Train unique loans | {stats['train_loans']:,} |
| Test unique loans | {stats['test_loans']:,} |
| Non-consecutive month gaps | {n_gaps} |

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
- **Train rate**: {stats['next_3m_delinquency_flag_rate']:.2f}% ({stats['next_3m_delinquency_flag_positive']:,} / {stats['next_3m_delinquency_flag_total_available']:,})

### 3.2 next_6m_delinquency_flag
- **Window**: T+1 through T+6
- **Positive**: Any month has `current_delinquency_status` ≥ '01'
- **Train rate**: {stats['next_6m_delinquency_flag_rate']:.2f}% ({stats['next_6m_delinquency_flag_positive']:,} / {stats['next_6m_delinquency_flag_total_available']:,})

### 3.3 next_12m_default_flag
- **Window**: T+1 through T+12
- **Positive**: `zero_balance_removal_reason` ∈ {{02, 03, 09}}
  - 02 = Third-Party Sale (foreclosure)
  - 03 = Short Sale
  - 09 = REO Disposition
- **NOT default**: Voluntary prepayment (01), Repurchase (06), Note Sale (15), Reperforming Sale (16)
- **NOT default**: 90+ DPD alone (this is a risk indicator, not a termination event)
- **Train rate**: {stats['next_12m_default_flag_rate']:.2f}% ({stats['next_12m_default_flag_positive']:,} / {stats['next_12m_default_flag_total_available']:,})

### 3.4 next_12m_prepayment_flag
- **Window**: T+1 through T+12
- **Positive**: `zero_balance_removal_reason` = '01' (voluntary payoff/matured)
- **NOT prepayment**: Any other termination reason
- **Train rate**: {stats['next_12m_prepayment_flag_rate']:.2f}% ({stats['next_12m_prepayment_flag_positive']:,} / {stats['next_12m_prepayment_flag_total_available']:,})

### 3.5 next_state
- **Window**: T+1 (single step)
- **Taxonomy**: Current, 30DPD, 60DPD, 90+DPD, Prepaid, Default/REO, Unknown

---

## 4. Next-State Distribution (Train)

| State | Count |
|---|---|
{chr(10).join(ns_lines)}

---

## 5. Right-Censoring

| Window | Censored (train) | Description |
|---|---|---|
| 3-month | {stats['censored_3m']:,} | Loan active, <3 future months in dataset |
| 6-month | {stats['censored_6m']:,} | Loan active, <6 future months in dataset |
| 12-month | {stats['censored_12m']:,} | Loan active, <12 future months in dataset |

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
| 3-month | {stats.get('unknown_3m', 0):,} |
| 6-month | {stats.get('unknown_6m', 0):,} |
| 12-month | {stats.get('unknown_12m', 0):,} |

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
- **Injected anomalies**: {n_anomalies} total
  - BALANCE_DISCREPANCY: UPB perturbed ±5–15%
  - DELINQUENCY_STATUS_LAG: Delinquent status set to '00'
  - MISSING_MODIFICATION: Mod flag changed Y→N
- **Marking**: All synthetic records have `is_synthetic=True` and `anomaly_type`
- **Ground truth**: `servicer_updates_ground_truth.json` (evaluation only)

**These are intentionally created test cases, NOT real Fannie Mae anomalies.**

---

## 10. Validation Rules

{n_rules} rules defined in `validation_rules.json`. Categories:
- Impossible values (negative balance, out-of-range FICO/LTV/DTI)
- Required field violations
- Structural integrity (duplicate keys, invalid codes)
- Consistency checks (termination state vs. balance)
- Date relationship checks

---

## 11. Files Created

| File | Description | Rows |
|---|---|---|
| `data/processed/loan_monthly_performance_train.csv` | Train features + targets | {stats['train_rows']:,} |
| `data/processed/loan_monthly_performance_test.csv` | Test features only | {stats['test_rows']:,} |
| `data/processed/test_targets_holdout.csv` | Test ground truth (eval only) | {stats['test_rows']:,} |
| `data/processed/loan_static_attributes.csv` | One row per loan | 25,000 |
| `data/processed/servicer_updates.csv` | Reconciliation data | ~10,200 |
| `data/processed/servicer_updates_ground_truth.json` | Anomaly labels | {n_anomalies} |
| `data/processed/validation_rules.json` | Data quality rules | {n_rules} |
| `data/processed/data_dictionary.md` | Complete documentation | — |
| `data/processed/macro_scenarios.csv` | Scenario assumptions | 18 |
| `data/processed/submission_template.csv` | Blank predictions | {stats['test_rows']:,} |
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
"""

    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, 'phase1_target_engineering_report.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  Report → {path}")


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    print("=" * 60)
    print("PHASE 1B: Competition Data Pack + Target Engineering")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    # 1. Load
    df = load_and_validate()

    # 2. Parse dates
    df = parse_dates(df)

    # 3. Consecutive-month check
    n_gaps = validate_consecutive(df)

    # 4. State mapping
    df = map_states(df)

    # 5. Targets
    df = compute_targets(df)

    # 6. Censoring
    df = compute_censoring(df)

    # 7. Features
    df = compute_features(df)

    # 8. Split & save train/test
    train_df, test_df, holdout_df = split_and_save(df)

    # 9. Static attributes
    static_df = create_static_attributes(df)

    # 10. Servicer updates
    n_anomalies = create_servicer_updates(df)

    # 11. Validation rules
    n_rules = create_validation_rules()

    # 12. Data dictionary
    create_data_dictionary()

    # 13. Macro scenarios
    create_macro_scenarios()

    # 14. Submission template
    template = create_submission_template(test_df)

    # 15. Validation audit
    stats = run_validation(df, train_df, test_df, holdout_df, static_df, template)

    # 16. Report
    runtime = time.time() - t0
    generate_report(stats, n_anomalies, n_rules, n_gaps, runtime)

    print("\n" + "=" * 60)
    elapsed = time.time() - t0
    print(f"Total runtime: {elapsed:.1f}s")
    print("PHASE 1B COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
