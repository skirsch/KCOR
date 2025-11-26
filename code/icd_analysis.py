#!/usr/bin/env python3
"""
ICD-10 Cause of Death Analysis

Compare cause-of-death distributions between vaccinated and unvaccinated individuals
using ICD-10 codes from the Czech event-level dataset.

USAGE:
    python icd_analysis.py [input_csv] [output_dir]
    
    Defaults:
    - input_csv: ../data/Czech2/data.csv
    - output_dir: ../data/Czech2/

OUTPUTS:
    - icd_comparison.csv: Main comparison table with ICD distributions
    - icd_difference_plot.png: Visualization of top differences
    - icd_by_system.csv: Aggregated comparison by organ system
    - icd_agegroup_*.csv: Age-stratified comparisons (one file per age group)
    - icd_postvax_bin_*.csv: Time-since-last-dose distributions (one file per time bin)
    - icd_by_dose_*.csv: Dose-specific ICD distributions (one file per dose count)
    - icd_summary.txt: Summary statistics

DEPENDENCIES:
    pip install pandas matplotlib
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Set, Optional


def map_icd_to_system(icd_code: str) -> str:
    """
    Map ICD-10 code to organ system category.
    
    Returns organ system name or "Other" if not categorized.
    """
    if pd.isna(icd_code) or not isinstance(icd_code, str) or len(icd_code) == 0:
        return "Unknown"
    
    icd_upper = icd_code.upper().strip()
    
    # Extract first 3 characters (category level)
    if len(icd_upper) >= 3:
        prefix = icd_upper[:3]
    else:
        prefix = icd_upper
    
    # Cardiovascular (I00-I99)
    if prefix.startswith('I'):
        # Myocarditis/pericarditis (I40-I42)
        if prefix in ['I40', 'I41', 'I42']:
            return "Myocarditis/Pericarditis"
        # Acute myocardial infarction (I21-I24)
        elif prefix in ['I21', 'I22', 'I23', 'I24']:
            return "Cardiovascular (AMI)"
        # Cardiac arrest (I46)
        elif prefix == 'I46':
            return "Cardiovascular (Cardiac Arrest)"
        # Cerebral infarction (I63)
        elif prefix == 'I63':
            return "Cardiovascular (Stroke)"
        # Other cardiovascular
        elif prefix[0] == 'I':
            return "Cardiovascular (Other)"
    
    # Neoplasms/Cancers (C00-C97)
    if prefix[0] == 'C':
        return "Cancer"
    
    # Respiratory (J00-J99)
    if prefix[0] == 'J':
        # Pneumonia (J12-J18)
        if prefix in ['J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18']:
            return "Respiratory (Pneumonia)"
        # Respiratory failure (J80)
        elif prefix == 'J80':
            return "Respiratory (Failure)"
        # Other respiratory
        else:
            return "Respiratory (Other)"
    
    # Neurologic (G00-G99)
    if prefix[0] == 'G':
        # Brain damage (G93)
        if prefix == 'G93':
            return "Neurologic (Brain Damage)"
        else:
            return "Neurologic (Other)"
    
    # Renal (N00-N99)
    if prefix[0] == 'N':
        # Acute kidney failure (N17-N19)
        if prefix in ['N17', 'N18', 'N19']:
            return "Renal (Acute Failure)"
        else:
            return "Renal (Other)"
    
    # Infectious diseases (A00-B99)
    if prefix[0] in ['A', 'B']:
        return "Infectious Disease"
    
    # Endocrine (E00-E90)
    if prefix[0] == 'E':
        return "Endocrine"
    
    # Mental/Behavioral (F00-F99)
    if prefix[0] == 'F':
        return "Mental/Behavioral"
    
    # External causes (V00-Y99)
    if prefix[0] in ['V', 'W', 'X', 'Y']:
        return "External Causes"
    
    # Other
    return "Other"


def calculate_age(birth_year, event_year, event_month) -> Optional[float]:
    """Calculate age from birth year and event date."""
    # Convert to numeric if needed
    try:
        birth_year = pd.to_numeric(birth_year, errors='coerce')
        event_year = pd.to_numeric(event_year, errors='coerce')
        event_month = pd.to_numeric(event_month, errors='coerce')
    except (TypeError, ValueError):
        return None
    
    if pd.isna(birth_year) or pd.isna(event_year):
        return None
    
    # Approximate age using year and month
    age = float(event_year) - float(birth_year)
    if not pd.isna(event_month):
        # Adjust for month (rough approximation)
        age += (float(event_month) - 1) / 12.0
    
    return age


def age_to_group(age: Optional[float]) -> str:
    """Convert age to age group."""
    if age is None or pd.isna(age):
        return "Unknown"
    elif age < 40:
        return "0-39"
    elif age < 60:
        return "40-59"
    elif age < 70:
        return "60-69"
    elif age < 80:
        return "70-79"
    elif age < 90:
        return "80-89"
    else:
        return "90+"


def main():
    # Parse command-line arguments
    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        output_dir = sys.argv[2]
    elif len(sys.argv) == 2:
        input_file = sys.argv[1]
        output_dir = "../data/Czech2/"
    else:
        input_file = "../data/Czech2/data.csv"
        output_dir = "../data/Czech2/"
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"[ICD Analysis] Reading input file: {input_file}")
    print(f"[ICD Analysis] Output directory: {output_dir}")
    
    # Step 1: Load and clean data
    print("\n[Step 1] Loading and cleaning data...")
    try:
        df = pd.read_csv(input_file, dtype=str, low_memory=False)
        print(f"  Loaded {len(df):,} records")
    except FileNotFoundError:
        print(f"ERROR: Input file not found: {input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to read input file: {e}")
        sys.exit(1)
    
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    print(f"  Columns: {len(df.columns)}")
    
    # Convert numeric fields where appropriate
    numeric_cols = ["id_zeny", "rok_udalosti", "mesic_udalosti",
                    "covid_ocko_poradi_davky", "covid_onemoc_poradi_infekce"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            print(f"  Converted {c} to numeric")
    
    # Step 2: Identify death records
    print("\n[Step 2] Identifying death records...")
    if "udalost" not in df.columns:
        print("ERROR: Column 'udalost' not found in data")
        sys.exit(1)
    
    death_df = df[df["udalost"].str.lower() == "umrti"].copy()
    print(f"  Found {len(death_df):,} death records")
    
    if len(death_df) == 0:
        print("ERROR: No death records found")
        sys.exit(1)
    
    # Identify ICD column
    icd_cols = [c for c in death_df.columns if "diag" in c]
    if not icd_cols:
        print("ERROR: No ICD diagnosis column found (expected column containing 'diag')")
        sys.exit(1)
    
    icd_col = icd_cols[0]
    print(f"  Using ICD column: {icd_col}")
    
    death_df["icd"] = death_df[icd_col].str.strip()
    death_df = death_df[death_df["icd"].notna() & (death_df["icd"] != "")]
    print(f"  Death records with ICD codes: {len(death_df):,}")
    
    # Step 3: Determine vaccination status
    print("\n[Step 3] Determining vaccination status...")
    vaccinated_ids = set(
        df[df["udalost"].str.lower() == "covid ockovani"]["id_zeny"].unique()
    )
    print(f"  Found {len(vaccinated_ids):,} vaccinated individuals")
    
    death_df["vaccinated"] = death_df["id_zeny"].isin(vaccinated_ids)
    vaccinated_deaths = death_df["vaccinated"].sum()
    unvaccinated_deaths = (~death_df["vaccinated"]).sum()
    print(f"  Vaccinated deaths: {vaccinated_deaths:,}")
    print(f"  Unvaccinated deaths: {unvaccinated_deaths:,}")
    
    # Step 4: Build ICD histograms
    print("\n[Step 4] Building ICD histograms...")
    vaccinated_counts = (
        death_df[death_df["vaccinated"] == True]
        .groupby("icd")
        .size()
        .sort_values(ascending=False)
    )
    print(f"  Unique ICD codes (vaccinated): {len(vaccinated_counts)}")
    
    unvaccinated_counts = (
        death_df[death_df["vaccinated"] == False]
        .groupby("icd")
        .size()
        .sort_values(ascending=False)
    )
    print(f"  Unique ICD codes (unvaccinated): {len(unvaccinated_counts)}")
    
    # Step 5: Normalize distributions
    print("\n[Step 5] Normalizing distributions...")
    vaccinated_dist = vaccinated_counts / vaccinated_counts.sum()
    unvaccinated_dist = unvaccinated_counts / unvaccinated_counts.sum()
    
    # Step 6: Create comparison table
    print("\n[Step 6] Creating comparison table...")
    comparison = pd.DataFrame({
        "vaccinated_count": vaccinated_counts,
        "vaccinated_pct": vaccinated_dist,
        "unvaccinated_count": unvaccinated_counts,
        "unvaccinated_pct": unvaccinated_dist
    }).fillna(0)
    
    comparison["difference"] = comparison["vaccinated_pct"] - comparison["unvaccinated_pct"]
    comparison["abs_difference"] = comparison["difference"].abs()
    comparison.sort_values("abs_difference", ascending=False, inplace=True)
    
    output_file = output_path / "icd_comparison.csv"
    comparison.to_csv(output_file)
    print(f"  Saved: {output_file}")
    
    # Step 7: Plot differences
    print("\n[Step 7] Creating difference plot...")
    top_n = 20
    # Get top N by absolute difference (already sorted by abs_difference)
    top_icds = comparison.head(top_n)
    
    plt.figure(figsize=(12, 8))
    colors = ['red' if x > 0 else 'blue' for x in top_icds["difference"]]
    # Convert proportions to percentage points for clarity (multiply by 100)
    differences_pct = top_icds["difference"] * 100
    plt.barh(range(len(top_icds)), differences_pct, color=colors)
    plt.yticks(range(len(top_icds)), top_icds.index)
    plt.xlabel("Difference (percentage points): Vaccinated % - Unvaccinated %")
    plt.title("Difference in Cause-of-Death Distribution (by ICD-10 Category)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    plot_file = output_path / "icd_difference_plot.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_file}")
    
    # Extension 8: ICD grouping by organ system (using lookup file)
    print("\n[Extension 8] Grouping ICDs by organ system...")
    lookup_file = output_path / "icd_system_lookup.csv"
    if lookup_file.exists():
        lookup = pd.read_csv(lookup_file)
        death_df_system = death_df.copy()
        death_df_system["icd_prefix"] = death_df_system["icd"].str[:1]
        death_df_system = death_df_system.merge(lookup, on="icd_prefix", how="left")
        death_df_system["system"] = death_df_system["system"].fillna("other")
        
        organ_dist = (
            death_df_system.groupby(["system", "vaccinated"])
            .size()
            .unstack(fill_value=0)
        )
        
        # Calculate percentages for each system
        system_comparison = pd.DataFrame({
            "vaccinated_count": organ_dist.get(True, pd.Series(dtype=int, index=organ_dist.index)),
            "unvaccinated_count": organ_dist.get(False, pd.Series(dtype=int, index=organ_dist.index))
        }).fillna(0)
        
        total_vacc = system_comparison["vaccinated_count"].sum()
        total_unvacc = system_comparison["unvaccinated_count"].sum()
        
        if total_vacc > 0:
            system_comparison["vaccinated_pct"] = system_comparison["vaccinated_count"] / total_vacc
        else:
            system_comparison["vaccinated_pct"] = 0
        
        if total_unvacc > 0:
            system_comparison["unvaccinated_pct"] = system_comparison["unvaccinated_count"] / total_unvacc
        else:
            system_comparison["unvaccinated_pct"] = 0
        
        system_comparison["difference"] = system_comparison["vaccinated_pct"] - system_comparison["unvaccinated_pct"]
        system_comparison["abs_difference"] = system_comparison["difference"].abs()
        system_comparison.sort_values("abs_difference", ascending=False, inplace=True)
        
        system_file = output_path / "icd_by_system.csv"
        system_comparison.to_csv(system_file)
        print(f"  Saved: {system_file}")
    else:
        print(f"  Warning: Lookup file not found: {lookup_file}, using fallback mapping")
        # Fallback to old method if lookup file doesn't exist
        death_df_system = death_df.copy()
        death_df_system["organ_system"] = death_df_system["icd"].apply(map_icd_to_system)
        
        system_vaccinated = (
            death_df_system[death_df_system["vaccinated"] == True]
            .groupby("organ_system")
            .size()
        )
        system_unvaccinated = (
            death_df_system[death_df_system["vaccinated"] == False]
            .groupby("organ_system")
            .size()
        )
        
        system_vaccinated_pct = system_vaccinated / system_vaccinated.sum()
        system_unvaccinated_pct = system_unvaccinated / system_unvaccinated.sum()
        
        system_comparison = pd.DataFrame({
            "vaccinated_count": system_vaccinated,
            "vaccinated_pct": system_vaccinated_pct,
            "unvaccinated_count": system_unvaccinated,
            "unvaccinated_pct": system_unvaccinated_pct
        }).fillna(0)
        
        system_comparison["difference"] = system_comparison["vaccinated_pct"] - system_comparison["unvaccinated_pct"]
        system_comparison["abs_difference"] = system_comparison["difference"].abs()
        system_comparison.sort_values("abs_difference", ascending=False, inplace=True)
        
        system_file = output_path / "icd_by_system.csv"
        system_comparison.to_csv(system_file)
        print(f"  Saved: {system_file}")
    
    # Extension 9: Age-stratified ICD comparisons
    print("\n[Extension 9] Creating age-stratified comparisons...")
    # Calculate age if we have birth year and event year
    if "rok_narozeni" in death_df.columns and "rok_udalosti" in death_df.columns:
        # Simple age calculation: rok_udalosti - rok_narozeni
        death_df["age"] = pd.to_numeric(death_df["rok_udalosti"], errors="coerce") - pd.to_numeric(death_df["rok_narozeni"], errors="coerce")
        
        # Define age bins: [0,40,60,70,80,90,200] with labels ["0-39","40-59","60-69","70-79","80-89","90+"]
        bins = [0, 40, 60, 70, 80, 90, 200]
        labels = ["0-39", "40-59", "60-69", "70-79", "80-89", "90+"]
        
        death_df["age_group"] = pd.cut(death_df["age"], bins=bins, labels=labels, right=False)
        
        age_results = []
        for age_group in labels:
            age_deaths = death_df[death_df["age_group"] == age_group]
            if len(age_deaths) == 0:
                continue
            
            vacc_sub = age_deaths[age_deaths["vaccinated"] == True]
            unvacc_sub = age_deaths[age_deaths["vaccinated"] == False]
            
            if len(vacc_sub) == 0 or len(unvacc_sub) == 0:
                continue
            
            vacc_counts = vacc_sub.groupby("icd").size()
            unvacc_counts = unvacc_sub.groupby("icd").size()
            
            vacc_pct = vacc_counts / len(vacc_sub)
            unvacc_pct = unvacc_counts / len(unvacc_sub)
            
            combined = pd.DataFrame({
                "vacc_pct": vacc_pct,
                "unvacc_pct": unvacc_pct
            }).fillna(0)
            
            combined["diff"] = combined["vacc_pct"] - combined["unvacc_pct"]
            combined.sort_values("diff", ascending=False, inplace=True)
            
            age_file = output_path / f"icd_agegroup_{age_group}.csv"
            combined.to_csv(age_file)
            print(f"  Saved: {age_file}")
            age_results.append(age_group)
        
        if not age_results:
            print("  Warning: No age data available for stratification")
    else:
        print("  Warning: Missing birth year or event year columns for age stratification")
    
    # Extension 10: Time-Since-Last-Dose Analysis
    print("\n[Extension 10] Analyzing time since last dose...")
    if "rok_udalosti" in df.columns and "mesic_udalosti" in df.columns:
        # Extract vaccine events
        vax_events = df[df["udalost"].str.lower() == "covid ockovani"][
            ["id_zeny", "rok_udalosti", "mesic_udalosti", "covid_ocko_poradi_davky"]
        ].copy()
        
        if len(vax_events) > 0:
            # Convert to numeric
            vax_events["rok_udalosti"] = pd.to_numeric(vax_events["rok_udalosti"], errors="coerce")
            vax_events["mesic_udalosti"] = pd.to_numeric(vax_events["mesic_udalosti"], errors="coerce")
            vax_events["covid_ocko_poradi_davky"] = pd.to_numeric(vax_events["covid_ocko_poradi_davky"], errors="coerce")
            
            # Sort to get last dose per person
            vax_events = vax_events.sort_values(
                ["id_zeny", "rok_udalosti", "mesic_udalosti", "covid_ocko_poradi_davky"]
            )
            last_vax = vax_events.groupby("id_zeny").last().reset_index()
            last_vax = last_vax.rename(columns={
                "rok_udalosti": "rok_udalosti_vax",
                "mesic_udalosti": "mesic_udalosti_vax"
            })
            
            # Merge with death records
            death_df_vax = death_df.merge(
                last_vax[["id_zeny", "rok_udalosti_vax", "mesic_udalosti_vax"]],
                on="id_zeny",
                how="left"
            )
            
            # Calculate months since last dose
            death_df_vax["rok_udalosti"] = pd.to_numeric(death_df_vax["rok_udalosti"], errors="coerce")
            death_df_vax["mesic_udalosti"] = pd.to_numeric(death_df_vax["mesic_udalosti"], errors="coerce")
            
            death_df_vax["months_since_dose"] = (
                (death_df_vax["rok_udalosti"] - death_df_vax["rok_udalosti_vax"]) * 12 +
                (death_df_vax["mesic_udalosti"] - death_df_vax["mesic_udalosti_vax"])
            )
            
            # Bin time since dose: bins = [-999,0,2,4,7,13,1000], labels = ["0-0","1-2","3-4","5-7","8-12","12+"]
            bins = [-999, 0, 2, 4, 7, 13, 1000]
            labels = ["0-0", "1-2", "3-4", "5-7", "8-12", "12+"]
            
            death_df_vax["post_vax_bin"] = pd.cut(
                death_df_vax["months_since_dose"],
                bins=bins,
                labels=labels
            )
            
            # Compute ICD distribution per bin (vaccinated deaths only)
            for b in labels:
                sub = death_df_vax[
                    (death_df_vax["post_vax_bin"] == b) &
                    (death_df_vax["vaccinated"] == True)
                ]
                
                if len(sub) == 0:
                    continue
                
                icd_counts = sub.groupby("icd").size().sort_values(ascending=False)
                icd_pct = icd_counts / icd_counts.sum()
                
                icd_pct_df = pd.DataFrame({
                    "count": icd_counts,
                    "pct": icd_pct
                })
                
                bin_file = output_path / f"icd_postvax_bin_{b}.csv"
                icd_pct_df.to_csv(bin_file)
                print(f"  Saved: {bin_file}")
        else:
            print("  Warning: No vaccination events found")
    else:
        print("  Warning: Missing date columns for time-since-dose analysis")
    
    # Extension 11: Dose-Specific ICD Comparison
    print("\n[Extension 11] Analyzing dose-specific ICD distributions...")
    if "covid_ocko_poradi_davky" in df.columns:
        # Get maximum dose count per person from vaccination events
        dose_groups = df[df["udalost"].str.lower() == "covid ockovani"].groupby("id_zeny")["covid_ocko_poradi_davky"].max()
        dose_groups = pd.to_numeric(dose_groups, errors="coerce")
        
        # Merge dose count into death records
        death_df_dose = death_df.merge(
            dose_groups.reset_index(name="dose_count"),
            on="id_zeny",
            how="left"
        )
        
        # For each dose count, compute ICD distribution
        unique_doses = sorted(death_df_dose["dose_count"].dropna().unique())
        
        for d in unique_doses:
            if pd.isna(d):
                continue
            
            sub = death_df_dose[death_df_dose["dose_count"] == d]
            
            if len(sub) == 0:
                continue
            
            icd_counts = sub.groupby("icd").size().sort_values(ascending=False)
            icd_pct = icd_counts / icd_counts.sum()
            
            icd_pct_df = pd.DataFrame({
                "count": icd_counts,
                "pct": icd_pct
            })
            
            dose_file = output_path / f"icd_by_dose_{int(d)}.csv"
            icd_pct_df.to_csv(dose_file)
            print(f"  Saved: {dose_file}")
    else:
        print("  Warning: Missing dose count column for dose-specific analysis")
    
    # Extension 12: Summary statistics
    print("\n[Extension 12] Generating summary statistics...")
    summary_lines = [
        "ICD-10 Cause of Death Analysis Summary",
        "=" * 50,
        "",
        f"Total death records analyzed: {len(death_df):,}",
        f"  Vaccinated deaths: {vaccinated_deaths:,} ({100*vaccinated_deaths/len(death_df):.1f}%)",
        f"  Unvaccinated deaths: {unvaccinated_deaths:,} ({100*unvaccinated_deaths/len(death_df):.1f}%)",
        "",
        f"Unique ICD-10 codes: {len(comparison):,}",
        "",
        "Top 10 ICD codes (vaccinated deaths):",
    ]
    
    top_vaccinated = comparison.sort_values("vaccinated_count", ascending=False).head(10)
    for idx, (icd, row) in enumerate(top_vaccinated.iterrows(), 1):
        summary_lines.append(f"  {idx:2d}. {icd}: {row['vaccinated_count']:.0f} ({100*row['vaccinated_pct']:.2f}%)")
    
    summary_lines.extend([
        "",
        "Top 10 ICD codes (unvaccinated deaths):",
    ])
    
    top_unvaccinated = comparison.sort_values("unvaccinated_count", ascending=False).head(10)
    for idx, (icd, row) in enumerate(top_unvaccinated.iterrows(), 1):
        summary_lines.append(f"  {idx:2d}. {icd}: {row['unvaccinated_count']:.0f} ({100*row['unvaccinated_pct']:.2f}%)")
    
    summary_lines.extend([
        "",
        "Top 10 ICD codes with largest differences (vaccinated - unvaccinated):",
    ])
    
    top_diff = comparison.head(10)
    for idx, (icd, row) in enumerate(top_diff.iterrows(), 1):
        summary_lines.append(f"  {idx:2d}. {icd}: {100*row['difference']:+.2f}% (vacc: {100*row['vaccinated_pct']:.2f}%, unvacc: {100*row['unvaccinated_pct']:.2f}%)")
    
    summary_lines.extend([
        "",
        "Organ system distribution:",
    ])
    
    for system, row in system_comparison.iterrows():
        summary_lines.append(f"  {system}: vacc {100*row['vaccinated_pct']:.1f}% vs unvacc {100*row['unvaccinated_pct']:.1f}% (diff: {100*row['difference']:+.1f}%)")
    
    summary_text = "\n".join(summary_lines)
    summary_file = output_path / "icd_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"  Saved: {summary_file}")
    
    print("\n[ICD Analysis] Analysis complete!")
    print(f"\nOutput files:")
    print(f"  - {output_path / 'icd_comparison.csv'}")
    print(f"  - {output_path / 'icd_difference_plot.png'}")
    print(f"  - {output_path / 'icd_by_system.csv'}")
    print(f"  - {output_path / 'icd_agegroup_*.csv'}")
    print(f"  - {output_path / 'icd_postvax_bin_*.csv'}")
    print(f"  - {output_path / 'icd_by_dose_*.csv'}")
    print(f"  - {output_path / 'icd_summary.txt'}")


if __name__ == "__main__":
    main()

