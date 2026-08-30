import zipfile
import os
import sys

raw_dir = r"c:\antigravity\Intain-Loan-Intelligence\data\raw"
archives = ["2009Q3.zip", "2009Q4.zip", "2010Q1.zip"]

all_unique_loans = set()
quarter_stats = {}

print("=== STREAMING INGESTION & LOAN POPULATION AUDIT ===")

for arch in archives:
    fpath = os.path.join(raw_dir, arch)
    if not os.path.exists(fpath):
        print(f"Error: {arch} not found!")
        continue
    
    q_loans = set()
    total_rows = 0
    malformed_rows = 0
    min_date = "999999"
    max_date = "000000"
    
    with zipfile.ZipFile(fpath, 'r') as z:
        csv_name = z.namelist()[0]
        print(f"\nProcessing {arch} -> {csv_name}...")
        with z.open(csv_name, 'r') as f:
            for idx, line_bytes in enumerate(f):
                line = line_bytes.decode('utf-8', errors='replace').rstrip('\r\n')
                if not line:
                    continue
                total_rows += 1
                parts = line.split('|')
                if len(parts) != 113:
                    malformed_rows += 1
                    continue
                
                loan_id = parts[1]
                rep_period = parts[2]
                
                q_loans.add(loan_id)
                all_unique_loans.add(loan_id)
                
                if rep_period < min_date and len(rep_period) == 6:
                    min_date = rep_period
                if rep_period > max_date and len(rep_period) == 6:
                    max_date = rep_period
                
                if total_rows % 5000000 == 0:
                    print(f"  ... processed {total_rows:,} rows, {len(q_loans):,} unique loans so far...")

    quarter_stats[arch] = {
        'total_rows': total_rows,
        'valid_rows': total_rows - malformed_rows,
        'malformed_rows': malformed_rows,
        'unique_loans': len(q_loans),
        'min_date': min_date,
        'max_date': max_date
    }
    print(f"Finished {arch}: {total_rows:,} rows | {len(q_loans):,} unique loans | Min period: {min_date} | Max period: {max_date}")

print("\n=== COMBINED POPULATION SUMMARY ===")
print(f"Total Unique Loans across all 3 archives: {len(all_unique_loans):,}")
for q, stats in quarter_stats.items():
    print(f"{q}: {stats['total_rows']:,} rows | {stats['valid_rows']:,} valid | {stats['malformed_rows']} malformed | {stats['unique_loans']:,} loans")

