import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Optional, List, Tuple
import argparse
import re

### Compute quantile-regression multipliers for ranges in Excel workbook using pandas
## For each sheet in the workbook, read cell C3 and D3 to get ranges (e.g. "Sheet1!A1:A10").
## Each range should be a single column of positive numbers.
## Compute the τ=10% quantile-regression multiplier for each range.
## Write the multiplier below the range on the CURRENT sheet.  

## Usage: python kcor_quantile_multipliers.py ../analysis/KCORv3.xlsx -o ../analysis/KCORv3_with_multipliers.xlsx --tau 0.10

def parse_range(a1: str):
    """Parse Excel range like 'Sheet1!A1:A10' or 'A1:A10'"""
    a1 = str(a1).strip().lower()  # Convert to lowercase to handle "c12:c226"
    if "!" in a1:
        sheet, r = a1.split("!", 1)
        sheet = sheet.strip().strip("'").strip('"')
    else:
        sheet, r = None, a1
    
    # Simple regex to parse range like A1:A10 (now case-insensitive)
    match = re.match(r'([a-z]+)(\d+):([a-z]+)(\d+)', r.strip())
    if not match:
        raise ValueError(f"Cannot parse range: {r}")
    
    col1, row1, col2, row2 = match.groups()
    
    # Convert column letters to numbers (A=1, B=2, etc.)
    def col_to_num(col):
        result = 0
        for char in col.upper():  # Convert back to uppercase
            result = result * 26 + (ord(char) - ord('A') + 1)
        return result
    
    c1, r1, c2, r2 = col_to_num(col1), int(row1), col_to_num(col2), int(row2)
    return sheet, (c1, r1, c2, r2)

def quantile_multiplier(values: List[float], tau: float = 0.10):
    y = np.asarray(values, dtype=float)
    if np.any(y <= 0) or y.size < 2:
        raise ValueError("All values must be > 0 and length >= 2.")
    t = np.arange(y.size, dtype=float)
    logy = np.log(y)
    X = sm.add_constant(t)
    res = sm.QuantReg(logy, X).fit(q=tau)
    b = float(res.params[1])  # slope coefficient
    # Return the correction multiplier (inverse of the trend)
    m = float(np.exp(-b))  # Note the negative sign - this is the correction factor
    return m, b

def process_workbook(input_path: str, output_path: Optional[str] = None, tau: float = 0.10):
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = base + "_with_multipliers" + ext

    print(f"Loading workbook with pandas: {input_path}")
    # Read all sheets - pandas handles pivot tables much better
    # Don't treat first row as header so Excel row numbering matches pandas indexing
    all_sheets = pd.read_excel(input_path, sheet_name=None, engine='openpyxl', header=None)
    print(f"Successfully loaded {len(all_sheets)} sheets")
    
    results = []
    
    # Create output workbook
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet_name, df in all_sheets.items():
            # Make a copy of the dataframe for modifications
            df_out = df.copy()
            sheet_has_multipliers = False  # Track if this sheet gets any multipliers
            
            # Check EXACTLY cells C3 and D3 (row index 2, column indices 2 and 3)
            for col_idx, col_name in enumerate(['C3', 'D3']):
                try:
                    # C3 = row index 2, col index 2; D3 = row index 2, col index 3
                    actual_col_idx = col_idx + 2  # C=2, D=3
                    
                    # Make sure we have enough rows and columns
                    if len(df) <= 2 or len(df.columns) <= actual_col_idx:
                        continue
                    
                    addr_val = df.iloc[2, actual_col_idx]  # Row 3 (index 2), Col C or D
                    
                    if pd.isna(addr_val):
                        continue
                    
                    # Convert to string and check if it looks like a range
                    addr_str = str(addr_val).strip()
                    if not addr_str or ':' not in addr_str:
                        continue
                    
                    # Only print once we find a valid range spec
                    if not sheet_has_multipliers:  # First valid range found in this sheet
                        print(f"Processing sheet: {sheet_name}")
                        print(f"Sheet has {len(df)} rows and {len(df.columns)} columns")
                        
                    print(f"  Found range spec in {col_name}: {addr_str}")
                    
                    try:
                        target_sheet, (c1, r1, c2, r2) = parse_range(addr_str)
                        print(f"    Parsed: sheet='{target_sheet}', range=({c1},{r1},{c2},{r2})")
                        
                        if c1 != c2:
                            print(f"    Range spans multiple columns; skipping")
                            continue

                        # Get the target dataframe
                        if target_sheet and target_sheet in all_sheets:
                            target_df = all_sheets[target_sheet]
                        else:
                            target_df = df

                        # Extract values from the range
                        start_row, end_row = r1 - 1, r2 - 1
                        target_col = c1 - 1
                        
                        if target_col >= len(target_df.columns):
                            print(f"    Column {c1} not found, skipping")
                            continue
                            
                        vals = []
                        for row_idx in range(start_row, min(end_row + 1, len(target_df))):
                            if row_idx >= len(target_df):
                                break
                            val = target_df.iloc[row_idx, target_col]
                            if pd.notna(val) and isinstance(val, (int, float)) and val > 0:
                                vals.append(float(val))

                        if len(vals) < 2:
                            print(f"    Not enough valid values (found {len(vals)}), skipping")
                            continue

                        print(f"    Computing multiplier for {len(vals)} values")
                        m, b = quantile_multiplier(vals, tau=tau)

                        # Write multiplier to C4 or D4
                        dest_row = 3  # Row 4 in Excel (index 3)
                        dest_col = actual_col_idx  # Same column as the range spec
                        
                        # Extend dataframe if necessary
                        while len(df_out) <= dest_row:
                            df_out.loc[len(df_out)] = [None] * len(df_out.columns)
                        
                        while len(df_out.columns) <= dest_col:
                            df_out[len(df_out.columns)] = None
                            
                        df_out.iloc[dest_row, dest_col] = round(m, 6)

                        dest_addr = f"{chr(65+dest_col)}{dest_row+1}"
                        results.append((sheet_name, addr_str, dest_addr, m, b))
                        print(f"    Written multiplier {m:.6f} to {dest_addr}")
                        sheet_has_multipliers = True
                        
                    except Exception as e:
                        print(f"    Parse failed: {e}")

                except Exception as e:
                    print(f"    Error: {e}")
                    continue
            
            # Only write sheets that have multipliers added
            if sheet_has_multipliers:
                print(f"  → Writing sheet to output")
                df_out.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
                print()  # Add blank line after processing sheet with multipliers
    
    return output_path, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute τ=10%% quantile-regression multipliers for ranges listed in C3 and D3 of each sheet; write result below each range.")
    parser.add_argument("input", help="Path to input .xlsx workbook")
    parser.add_argument("-o", "--output", help="Output path (.xlsx). Default: <input>_with_multipliers.xlsx", default=None)
    parser.add_argument("--tau", type=float, default=0.10, help="Quantile (default 0.10)")
    args = parser.parse_args()

    out, res = process_workbook(args.input, args.output, tau=args.tau)
    print(f"\nOutput written to: {out}")
    print("\nResults:")
    for sheet, addr, dest, m, b in res:
        print(f"{sheet}: {addr} -> {dest}  multiplier={m:.6f}  slope={b:+.8f}")