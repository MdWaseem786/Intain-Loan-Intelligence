import pandas as pd

df = pd.read_parquet('data/processed/selected_loan_performance.parquet')

print('=== SHAPE ===')
print(df.shape)
print()

print('=== COLUMNS & DTYPES ===')
for c in df.columns:
    print(f'  {c}: {df[c].dtype}')
print()

print('=== FIRST 3 ROWS (transposed) ===')
print(df.head(3).T.to_string())
print()

print('=== current_delinquency_status unique values ===')
vals = sorted(df['current_delinquency_status'].dropna().unique())
print(vals)
print()

print('=== zero_balance_removal_reason unique values ===')
vals = sorted(df['zero_balance_removal_reason'].dropna().unique())
print(vals)
print()

print('=== monthly_modification_flag unique values ===')
vals = sorted(df['monthly_modification_flag'].dropna().unique())
print(vals)
print()

mrp = df['monthly_reporting_period']
print('=== monthly_reporting_period range ===')
print(f'Min: {mrp.min()}, Max: {mrp.max()}')
print(f'Unique periods: {mrp.nunique()}')
print()

print('=== loan_id unique count ===')
print(df['loan_id'].nunique())
print()

print('=== Missing % for key cols ===')
key_cols = [
    'current_delinquency_status', 'zero_balance_removal_reason',
    'zero_balance_effective_date', 'monthly_modification_flag',
    'current_actual_upb', 'borrower_credit_score',
    'original_ltv', 'dti_ratio', 'original_interest_rate',
    'original_upb', 'original_loan_term', 'loan_age',
    'remaining_months_to_maturity', 'channel', 'loan_purpose',
    'property_type', 'occupancy_status', 'number_of_units',
    'property_state', 'servicer_name', 'seller_name'
]
for c in key_cols:
    miss = df[c].isna().mean() * 100
    print(f'  {c}: {miss:.2f}%')
print()

print('=== covid fields ===')
for c in ['covid_assistance_code', 'covid_forbearance_flag', 'covid_plan_end_date', 'covid_exit_reason']:
    if c in df.columns:
        vals = df[c].dropna().unique()
        print(f'  {c}: nunique={len(vals)}, samples={list(vals[:5])}')
print()

print('=== Sample reporting periods (sorted, first 10 and last 10) ===')
periods = sorted(mrp.unique())
print(f'First 10: {periods[:10]}')
print(f'Last 10: {periods[-10:]}')
print()

# Check for any columns beyond what's in the report
all_cols = list(df.columns)
print(f'=== Total columns: {len(all_cols)} ===')
# Print columns not in the report
cols_in_report = [
    'loan_id', 'monthly_reporting_period', 'channel', 'seller_name',
    'servicer_name', 'master_servicer', 'original_interest_rate',
    'current_interest_rate', 'original_upb', 'upb_at_issuance',
    'current_actual_upb', 'original_loan_term', 'origination_date',
    'first_payment_date', 'loan_age', 'remaining_months_to_maturity',
    'adjusted_months_to_maturity', 'maturity_date', 'original_ltv',
    'original_cltv', 'number_of_borrowers', 'dti_ratio',
    'borrower_credit_score', 'co_borrower_credit_score',
    'first_time_homebuyer_flag', 'loan_purpose', 'property_type',
    'number_of_units', 'occupancy_status', 'property_state', 'msa',
    'zip_code_short', 'mi_percentage', 'product_type',
    'special_eligibility_program', 'relocation_mortgage_flag',
    'current_delinquency_status', 'monthly_modification_flag',
    'zero_balance_removal_reason', 'zero_balance_effective_date',
    'upb_at_liquidation', 'covid_assistance_code',
    'covid_forbearance_flag', 'covid_plan_end_date', 'covid_exit_reason'
]
extra = [c for c in all_cols if c not in cols_in_report]
if extra:
    print(f'Columns NOT in report: {extra}')
else:
    print('All columns accounted for in Phase 1A report.')
