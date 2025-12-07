import pandas as pd
import numpy as np

# Read the CSV
df = pd.read_csv('data/Czech/KCOR_slope_debug.csv')

print("=" * 80)
print("ANALYSIS OF KCOR_slope_debug.csv")
print("=" * 80)

print(f"\nTotal rows: {len(df)}")
print(f"Unique enrollment dates: {df['enrollment_date'].nunique()}")
print(f"Unique modes: {df['mode'].unique()}")

# Filter to slope8 mode only (the optimization mode)
slope8_df = df[df['mode'] == 'slope8'].copy()
print(f"\nSlope8 rows: {len(slope8_df)}")

print("\n" + "=" * 80)
print("1. CASES WITH FEW DATA POINTS (< 15)")
print("=" * 80)
few_points = slope8_df[slope8_df['n_points'] < 15]
print(f"Count: {len(few_points)}")
if len(few_points) > 0:
    print("\nCases with < 15 points:")
    cols = ['enrollment_date', 'YearOfBirth', 'Dose', 'n_points', 'optimizer_success', 
            'optimizer_nfev', 'optimizer_nit', 'failure_detail']
    print(few_points[cols].to_string(index=False))

print("\n" + "=" * 80)
print("2. TOO MANY ITERATIONS FAILURES")
print("=" * 80)
failures = slope8_df[slope8_df['failure_detail'].str.contains('too_many_iterations', na=False)]
print(f"Count: {len(failures)}")
if len(failures) > 0:
    print("\nFailure cases:")
    cols = ['enrollment_date', 'YearOfBirth', 'Dose', 'n_points', 'optimizer_nfev', 
            'optimizer_nit', 'ka', 'kb', 'delta_k_init', 'tau']
    print(failures[cols].to_string(index=False))
    print(f"\nFailure rate: {len(failures)/len(slope8_df)*100:.1f}%")

print("\n" + "=" * 80)
print("3. CASES WITH VERY SMALL ka/kb VALUES (< 1e-6)")
print("=" * 80)
small_ka = slope8_df[(slope8_df['ka'].abs() < 1e-6) & (slope8_df['ka'].notna())]
small_kb = slope8_df[(slope8_df['kb'].abs() < 1e-6) & (slope8_df['kb'].notna())]
print(f"Small ka count: {len(small_ka)}")
print(f"Small kb count: {len(small_kb)}")
if len(small_ka) > 0:
    print("\nCases with small ka:")
    print(small_ka[['enrollment_date', 'YearOfBirth', 'Dose', 'ka', 'kb', 'optimizer_success']].head(10).to_string(index=False))
if len(small_kb) > 0:
    print("\nCases with small kb:")
    print(small_kb[['enrollment_date', 'YearOfBirth', 'Dose', 'ka', 'kb', 'optimizer_success']].head(10).to_string(index=False))

print("\n" + "=" * 80)
print("4. HIGH FUNCTION EVALUATIONS (nfev > 300)")
print("=" * 80)
high_nfev = slope8_df[(slope8_df['optimizer_nfev'] > 300) & (slope8_df['optimizer_nfev'].notna())]
print(f"Count: {len(high_nfev)}")
if len(high_nfev) > 0:
    print("\nCases with high nfev:")
    cols = ['enrollment_date', 'YearOfBirth', 'Dose', 'n_points', 'optimizer_nfev', 
            'optimizer_nit', 'optimizer_success']
    print(high_nfev[cols].head(20).to_string(index=False))
    print(f"\nAverage nfev for high cases: {high_nfev['optimizer_nfev'].mean():.1f}")
    print(f"Max nfev: {high_nfev['optimizer_nfev'].max()}")

print("\n" + "=" * 80)
print("5. CASES WITH delta_k_init NEAR ZERO (< 0.001)")
print("=" * 80)
small_delta = slope8_df[(slope8_df['delta_k_init'].abs() < 0.001) & (slope8_df['delta_k_init'].notna())]
print(f"Count: {len(small_delta)}")
if len(small_delta) > 0:
    print("\nCases with small delta_k_init:")
    cols = ['enrollment_date', 'YearOfBirth', 'Dose', 'n_points', 'delta_k_init', 
            'tau', 'optimizer_success', 'optimizer_nfev', 'optimizer_nit']
    print(small_delta[cols].head(20).to_string(index=False))
    print(f"\nSuccess rate for small delta_k: {small_delta['optimizer_success'].sum()/len(small_delta)*100:.1f}%")
    print(f"Average nfev for small delta_k: {small_delta['optimizer_nfev'].mean():.1f}")

print("\n" + "=" * 80)
print("6. OVERALL SUCCESS RATE")
print("=" * 80)
success_rate = slope8_df['optimizer_success'].sum() / len(slope8_df) * 100
print(f"Success rate: {success_rate:.1f}%")
print(f"Total successful: {slope8_df['optimizer_success'].sum()}")
print(f"Total failed: {(~slope8_df['optimizer_success']).sum()}")

print("\n" + "=" * 80)
print("7. STATISTICS BY n_points")
print("=" * 80)
slope8_df['n_points_bin'] = pd.cut(slope8_df['n_points'], 
                                     bins=[0, 15, 30, 1000], 
                                     labels=['<15', '15-30', '>30'])
stats_by_points = slope8_df.groupby('n_points_bin').agg({
    'optimizer_success': ['count', 'sum', lambda x: x.sum()/len(x)*100],
    'optimizer_nfev': ['mean', 'max'],
    'optimizer_nit': ['mean', 'max']
}).round(2)
stats_by_points.columns = ['Total', 'Success', 'Success%', 'Avg_nfev', 'Max_nfev', 'Avg_nit', 'Max_nit']
print(stats_by_points)

print("\n" + "=" * 80)
print("8. TAU VALUES FOR SMALL delta_k CASES")
print("=" * 80)
if len(small_delta) > 0:
    print(f"Tau statistics for cases with delta_k < 0.001:")
    print(f"  Mean tau: {small_delta['tau'].mean():.2f}")
    print(f"  Median tau: {small_delta['tau'].median():.2f}")
    print(f"  Std tau: {small_delta['tau'].std():.2f}")
    print(f"  Min tau: {small_delta['tau'].min():.2f}")
    print(f"  Max tau: {small_delta['tau'].max():.2f}")
    print(f"\n  Cases with tau near 52 weeks (50-54): {((small_delta['tau'] >= 50) & (small_delta['tau'] <= 54)).sum()}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

