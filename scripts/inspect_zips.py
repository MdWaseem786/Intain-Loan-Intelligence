import zipfile
import os

raw_dir = r"c:\antigravity\Intain-Loan-Intelligence\data\raw"

for f in sorted(os.listdir(raw_dir)):
    if f.endswith('.zip'):
        fpath = os.path.join(raw_dir, f)
        with zipfile.ZipFile(fpath, 'r') as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name, 'r') as csv_file:
                first_line = csv_file.readline().decode('utf-8', errors='replace').rstrip('\r\n')
                second_line = csv_file.readline().decode('utf-8', errors='replace').rstrip('\r\n')
                print(f"=== {f} -> {csv_name} ===")
                print(f"  First line starts with '|': {first_line.startswith('|')}")
                fields_1 = first_line.split('|')
                fields_2 = second_line.split('|')
                print(f"  First line field count: {len(fields_1)}")
                print(f"  Second line field count: {len(fields_2)}")
                print(f"  Col 1 (Loan ID) sample L1: {fields_1[1] if len(fields_1)>1 else 'N/A'}")
                print(f"  Col 2 (Period) sample L1: {fields_1[2] if len(fields_1)>2 else 'N/A'}")
                print(f"  Col 1 (Loan ID) sample L2: {fields_2[1] if len(fields_2)>1 else 'N/A'}")
                print(f"  Col 2 (Period) sample L2: {fields_2[2] if len(fields_2)>2 else 'N/A'}")
