import pandas as pd

df = pd.read_parquet('data/processed/selected_loan_performance.parquet')

print('=== DELINQ STATUS FREQ (top 20) ===')
print(df['current_delinquency_status'].value_counts().head(20).to_string())
print()

print('=== ZERO BALANCE CODE FREQ ===')
print(df['zero_balance_removal_reason'].value_counts().to_string())
print()

# Parse reporting period into year/month
mrp = df['monthly_reporting_period']
# Format is MMYYYY
df['_year'] = mrp.str[-4:].astype(int)
df['_month'] = mrp.str[:2].astype(int)

train_mask = df['_year'] <= 2017
test_mask = df['_year'] >= 2018

print('=== TRAIN/TEST SPLIT ===')
print(f'Train rows (<=2017): {train_mask.sum()}')
print(f'Test rows (>=2018): {test_mask.sum()}')
print(f'Train unique loans: {df.loc[train_mask, "loan_id"].nunique()}')
print(f'Test unique loans: {df.loc[test_mask, "loan_id"].nunique()}')
print()

# ZB codes meaning for Fannie Mae:
# 01 = Prepaid or Matured
# 02 = Third Party Sale
# 03 = Short Sale
# 06 = Repurchase
# 09 = REO Disposition
# 15 = Note Sale
# 16 = Reperforming Loan Sale
print('=== ZERO BALANCE by train/test ===')
for mask, label in [(train_mask, 'Train'), (test_mask, 'Test')]:
    sub = df.loc[mask]
    zb = sub['zero_balance_removal_reason'].value_counts()
    print(f'{label}:')
    print(zb.to_string())
    print()

print('=== DELINQ STATUS in TRAIN ===')
dq_train = df.loc[train_mask, 'current_delinquency_status'].value_counts()
print(dq_train.head(15).to_string())
print()

print('=== Rows per year ===')
print(df.groupby('_year').size().to_string())

df.drop(columns=['_year', '_month'], inplace=True)
