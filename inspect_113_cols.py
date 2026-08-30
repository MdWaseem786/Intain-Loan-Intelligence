import zipfile
import pandas as pd
import io

fpath = r"c:\antigravity\Intain-Loan-Intelligence\data\raw\2009Q3.zip"

with zipfile.ZipFile(fpath, 'r') as z:
    csv_name = z.namelist()[0]
    with z.open(csv_name, 'r') as f:
        # Read first 100 lines into buffer
        lines = [f.readline().decode('utf-8', errors='replace').rstrip('\r\n') for _ in range(100)]

# Parse into dataframe
data = [line.split('|') for line in lines]
df = pd.DataFrame(data)

print(f"Shape of first 100 rows: {df.shape}")

for col in range(df.shape[1]):
    vals = df[col].replace('', None).dropna().tolist()
    sample = vals[0] if len(vals) > 0 else "<EMPTY>"
    print(f"Col {col:3d}: non_empty_in_100={len(vals):3d} | sample={sample}")
