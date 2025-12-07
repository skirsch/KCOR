import pandas as pd
import numpy as np
import sys

df = pd.read_csv('data/Czech/KCOR_slope_debug.csv')
slope8_df = df[df['mode'] == 'slope8'].copy()

# Redirect output to file
sys.stdout = open('fixes_analysis.txt', 'w')

print("=" * 80)
print("ANALYSIS: Did the fixes work?")
print("=" * 80)

# 1. Count failures
failures = slope8_df[slope8_df['failure_detail'].str.contains('too_many_iterations', na=False)]
successes = slope8_df[slope8_df['optimizer_success'] == True].copy()
failures_all = slope8_df[slope8_df['optimizer_success'] == False].copy()

print(f"\n1. OVERALL SUCCESS RATE")
print(f"   Total slope8 cases: {len(slope8_df)}")
print(f"   Successful: {len(successes)} ({len(successes)/len(slope8_df)*100:.1f}%)")
print(f"   Failed (too_many_iterations): {len(failures)} ({len(failures)/len(slope8_df)*100:.1f}%)")

# 2. Analyze failures
print(f"\n2. FAILURE ANALYSIS")
if len(failures) > 0:
    print(f"\n   Failed cases details:")
    for idx, row in failures.iterrows():
        print(f"\n   - {row['enrollment_date']}, YOB={row['YearOfBirth']}, Dose={row['Dose']}")
        print(f"     n_points={row['n_points']}, nfev={row['optimizer_nfev']}, nit={row['optimizer_nit']}")
        print(f"     delta_k_init={row['delta_k_init']:.6e}, ka={row['ka']:.6e}, kb={row['kb']:.6e}")
        print(f"     tau={row['tau']:.2f}, tau_init={row['tau_init']:.2f}")
        
    # Check if failures have small delta_k
    small_delta_failures = failures[(failures['delta_k_init'].abs() < 0.001) & (failures['delta_k_init'].notna())]
    print(f"\n   Failures with delta_k_init < 0.001: {len(small_delta_failures)}/{len(failures)}")
    
    # Check if failures have few points
    few_points_failures = failures[failures['n_points'] < 15]
    print(f"   Failures with n_points < 15: {len(few_points_failures)}/{len(failures)}")
    
    # Check tau values for failures
    print(f"\n   Tau values for failures:")
    print(f"     Mean: {failures['tau'].mean():.2f}")
    print(f"     Median: {failures['tau'].median():.2f}")
    print(f"     Range: [{failures['tau'].min():.2f}, {failures['tau'].max():.2f}]")
    print(f"     Cases with tau near 52 (50-54): {((failures['tau'] >= 50) & (failures['tau'] <= 54)).sum()}/{len(failures)}")

# 3. Check cases with small delta_k_init
print(f"\n3. CASES WITH SMALL delta_k_init (< 0.001)")
small_delta = slope8_df[(slope8_df['delta_k_init'].abs() < 0.001) & (slope8_df['delta_k_init'].notna())]
print(f"   Total cases: {len(small_delta)}")
if len(small_delta) > 0:
    small_delta_success = small_delta[small_delta['optimizer_success'] == True]
    print(f"   Successful: {len(small_delta_success)} ({len(small_delta_success)/len(small_delta)*100:.1f}%)")
    print(f"   Failed: {len(small_delta) - len(small_delta_success)} ({(len(small_delta) - len(small_delta_success))/len(small_delta)*100:.1f}%)")
    print(f"\n   Tau statistics for small delta_k cases:")
    print(f"     Mean: {small_delta['tau'].mean():.2f}")
    print(f"     Median: {small_delta['tau'].median():.2f}")
    print(f"     Cases with tau near 52 (50-54): {((small_delta['tau'] >= 50) & (small_delta['tau'] <= 54)).sum()}/{len(small_delta)}")
    print(f"     Average nfev: {small_delta['optimizer_nfev'].mean():.1f}")

# 4. Check cases with few data points
print(f"\n4. CASES WITH FEW DATA POINTS (< 15)")
few_points = slope8_df[slope8_df['n_points'] < 15]
print(f"   Total cases: {len(few_points)}")
if len(few_points) > 0:
    few_points_success = few_points[few_points['optimizer_success'] == True]
    print(f"   Successful: {len(few_points_success)} ({len(few_points_success)/len(few_points)*100:.1f}%)")
    print(f"   Failed: {len(few_points) - len(few_points_success)} ({(len(few_points) - len(few_points_success))/len(few_points)*100:.1f}%)")
    print(f"   Average nfev: {few_points['optimizer_nfev'].mean():.1f}")

# 5. Check high nfev cases
print(f"\n5. CASES WITH HIGH FUNCTION EVALUATIONS (nfev > 300)")
high_nfev = slope8_df[(slope8_df['optimizer_nfev'] > 300) & (slope8_df['optimizer_nfev'].notna())]
print(f"   Total cases: {len(high_nfev)}")
if len(high_nfev) > 0:
    high_nfev_success = high_nfev[high_nfev['optimizer_success'] == True]
    print(f"   Successful: {len(high_nfev_success)} ({len(high_nfev_success)/len(high_nfev)*100:.1f}%)")
    print(f"   Failed: {len(high_nfev) - len(high_nfev_success)}")
    print(f"   Average nfev: {high_nfev['optimizer_nfev'].mean():.1f}")
    print(f"   Max nfev: {high_nfev['optimizer_nfev'].max()}")

# 6. Statistics by n_points bins
print(f"\n6. SUCCESS RATE BY DATA POINT COUNT")
slope8_df['n_points_bin'] = pd.cut(slope8_df['n_points'], 
                                     bins=[0, 15, 30, 1000], 
                                     labels=['<15', '15-30', '>30'])
for bin_name in ['<15', '15-30', '>30']:
    bin_df = slope8_df[slope8_df['n_points_bin'] == bin_name]
    if len(bin_df) > 0:
        success_rate = bin_df['optimizer_success'].sum() / len(bin_df) * 100
        avg_nfev = bin_df['optimizer_nfev'].mean()
        print(f"   {bin_name:>5} points: {len(bin_df):3d} cases, {success_rate:5.1f}% success, avg nfev={avg_nfev:.0f}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"✓ Tau regularization: {'Working' if len(small_delta) > 0 and small_delta['tau'].median() > 40 else 'Check needed'}")
print(f"✓ Adaptive tolerances: {'Working' if len(few_points) > 0 and few_points['optimizer_success'].sum()/len(few_points) > 0.5 else 'Check needed'}")
print(f"✓ Overall improvement: {len(failures)} failures out of {len(slope8_df)} total cases ({len(failures)/len(slope8_df)*100:.1f}% failure rate)")

