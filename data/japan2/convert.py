import argparse
import csv
import datetime as dt
import glob
import os
import re
from typing import List, Optional, Tuple

import pandas as pd

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Japan2 vaccination CSV to KRF CSV")
    parser.add_argument("-c", "--config", default=None, help="Path to converter config YAML")
    parser.add_argument(
        "-i",
        "--input",
        required=False,
        help="Path to input CSV/CSV.XZ or directory containing .csv.xz files",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="krf.csv",
        help="Path to output KRF CSV (default: krf.csv)",
    )
    parser.add_argument(
        "--sidecar",
        default=None,
        help="Optional path to write sidecar YAML metadata (schema, observationEndDate, birthBandYears, timezone)",
    )
    parser.add_argument(
        "--observation-end",
        default=None,
        help="Observation end date YYYY-MM-DD for sidecar (optional)",
    )
    parser.add_argument(
        "--birth-band-years",
        type=int,
        default=5,
        choices=[1, 5, 10],
        help="Birth band width; if >1, YearOfBirth encodes band lower bound (default: 5)",
    )
    parser.add_argument(
        "--timezone",
        default="local",
        help="Timezone label to record in sidecar (default: local)",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Only include doses on/after this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Only include doses on/before this date (YYYY-MM-DD)",
    )
    return parser.parse_args()


def resolve_inputs(input_arg: str) -> List[str]:
    """Resolve input files - handle directory with .csv.xz files or single file."""
    if not input_arg:
        return []
    if os.path.isdir(input_arg):
        # Find all .csv.xz files in the directory
        files = sorted(glob.glob(os.path.join(input_arg, "*.csv.xz")))
        if not files:
            # Fall back to .csv files if no .xz files found
            files = sorted(glob.glob(os.path.join(input_arg, "*.csv")))
        if files:
            print(f"Found {len(files)} input file(s) in {input_arg}:", flush=True)
            for f in files:
                size = os.path.getsize(f) if os.path.exists(f) else 0
                size_mb = size / (1024 * 1024)
                print(f"  - {os.path.basename(f)} ({size_mb:.1f} MB)", flush=True)
        return files
    return [input_arg]


def read_concat(inputs: List[str]) -> pd.DataFrame:
    """Read and concatenate multiple CSV files, handling .xz compression."""
    frames: List[pd.DataFrame] = []
    total_files = len(inputs)
    for idx, path in enumerate(inputs, 1):
        try:
            file_basename = os.path.basename(path)
            print(f"[{idx}/{total_files}] Reading {file_basename}...", flush=True)
            
            # Handle .xz compressed files
            if path.endswith(".xz"):
                print(f"  Decompressing {file_basename}...", flush=True)
                # Try python lzma module first (more reliable)
                try:
                    import lzma
                    with lzma.open(path, 'rt', encoding='utf-8') as f:
                        df = pd.read_csv(
                            f,
                            header=0,
                            dtype=str,
                            na_values=["", "NULL", "null", "NaN"],
                            keep_default_na=False,
                            encoding="utf-8",
                            low_memory=False,
                        )
                    print(f"  Decompressed and read {file_basename}: {len(df)} rows", flush=True)
                except Exception as e_lzma:
                    print(f"  lzma module failed, trying xzcat subprocess...", flush=True)
                    # Fallback: use xzcat subprocess
                    import subprocess
                    proc = subprocess.Popen(["xzcat", path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    df = pd.read_csv(
                        proc.stdout,
                        header=0,
                        dtype=str,
                        na_values=["", "NULL", "null", "NaN"],
                        keep_default_na=False,
                        encoding="utf-8",
                        low_memory=False,
                    )
                    proc.wait()
                    if proc.returncode != 0:
                        stderr = proc.stderr.read().decode('utf-8', errors='ignore')
                        raise Exception(f"xzcat failed with return code {proc.returncode}: {stderr}")
                    print(f"  Decompressed and read {file_basename}: {len(df)} rows", flush=True)
            else:
                df = pd.read_csv(
                    path,
                    header=0,
                    dtype=str,
                    na_values=["", "NULL", "null", "NaN"],
                    keep_default_na=False,
                    encoding="utf-8",
                    low_memory=False,
                )
                print(f"  Read {file_basename}: {len(df)} rows", flush=True)
            frames.append(df)
        except Exception as e:
            print(f"ERROR: Failed to read {path}: {e}", flush=True)
            continue
    
    if not frames:
        print("ERROR: No files were successfully read!", flush=True)
        return pd.DataFrame()
    
    print(f"\nConcatenating {len(frames)} file(s)...", flush=True)
    combined = pd.concat(frames, axis=0, ignore_index=True)
    print(f"Total rows after concatenation: {len(combined)}", flush=True)
    return combined


def extract_age_band(age_str: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    """Extract age band bounds from strings like '00-09', '10-19', etc."""
    if not age_str:
        return None, None
    s = str(age_str).strip()
    try:
        # Format: "00-09", "10-19", "80-89", etc.
        if "-" in s:
            parts = s.split("-", 1)
            if len(parts) == 2:
                low = int(parts[0])
                high = int(parts[1])
                return low, high
        # Fallback: try to parse as single number
        n = int(s)
        return n, n
    except Exception:
        return None, None


def normalize_brand(value: Optional[str]) -> str:
    """Map manufacturer codes to KRF brand codes (P, M, O)."""
    if not value:
        return "O"
    v = str(value).strip().lower()
    mapping = {
        "pfizer": "P",
        "moderna": "M",
        "ファイザー": "P",
        "ﾌｧｲｻﾞｰ": "P",
        "ﾓﾃﾞﾙﾅ": "M",
        "モデルナ": "M",
        "第一三共": "O",
        "アストラゼネカ": "O",
        "武田": "O",
        "ノババックス": "O",
        "astrazeneca": "O",
        "novavax": "O",
        "other": "O",
    }
    return mapping.get(v, "O")


def parse_date(value: Optional[str]) -> Optional[dt.date]:
    """Parse date string to date object, handling ISO format."""
    if not value:
        return None
    v = str(value).strip()
    if not v or v.upper() == "NULL":
        return None
    try:
        dt_obj = dt.datetime.fromisoformat(v)
        return dt_obj.date()
    except Exception:
        try:
            dt_obj = dt.datetime.strptime(v, "%Y/%m/%d")
            return dt_obj.date()
        except Exception:
            return None


def normalize_date(value: Optional[object]) -> Optional[dt.date]:
    """Normalize YAML/CLI date inputs to date objects."""
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    v = str(value).strip()
    if not v:
        return None
    try:
        return dt.date.fromisoformat(v)
    except Exception:
        return None


def build_output(
    df: pd.DataFrame, start_date: Optional[object], end_date: Optional[object]
) -> pd.DataFrame:
    """Convert japan2 wide format to KRF format."""
    print(f"\nConverting {len(df)} records to KRF format...", flush=True)
    df = df.fillna("")

    # Preserve first-appearance order of each ID
    first_order = df.reset_index().groupby("id")["index"].min()

    # Parse date_age column (per-person reference date)
    date_age_series = pd.to_datetime(df.get("date_age", pd.Series([None] * len(df))), errors="coerce")

    # Extract age bands and compute YearOfBirth
    age_series = df.get("age", pd.Series([""] * len(df))).astype(str).str.strip()
    age_series = age_series.where(~age_series.str.upper().eq("NULL"), "")

    # Compute YearOfBirth per person using date_age as reference
    yob_list = []
    start_dt = normalize_date(start_date)
    end_dt = normalize_date(end_date)

    for idx in df.index:
        age_str = str(age_series.loc[idx]) if idx in age_series.index else ""
        ref_date = date_age_series.loc[idx] if idx in date_age_series.index else pd.NaT
        
        if pd.isna(ref_date) or not age_str:
            yob_list.append(None)
            continue
        
        low, high = extract_age_band(age_str)
        if high is None:
            yob_list.append(None)
            continue
        
        # Clamp open-ended to 100
        if high >= 130:
            high = 100
        
        # YearOfBirth = reference_year - age_upper_bound
        yob_raw = ref_date.year - int(high)
        # Bucket to 5-year cohort lower bound
        yob_list.append((yob_raw // 5) * 5)
    
    yob_series = pd.Series(yob_list, index=df.index, dtype="Int64")

    # Collect doses from wide format columns (date_dose1-9, pharma_dose1-9)
    dose_data = []
    for idx in df.index:
        person_id = str(df.loc[idx, "id"]).strip()
        doses = []
        for dose_num in range(1, 10):
            date_col = f"date_dose{dose_num}"
            pharma_col = f"pharma_dose{dose_num}"
            if date_col in df.columns and pharma_col in df.columns:
                date_val = df.loc[idx, date_col]
                pharma_val = df.loc[idx, pharma_col]
                if date_val and str(date_val).strip() and str(date_val).upper() != "NULL":
                    date_obj = parse_date(date_val)
                    if date_obj:
                        # Filter by date window if specified
                        if start_dt and date_obj < start_dt:
                            continue
                        if end_dt and date_obj > end_dt:
                            continue
                        brand = normalize_brand(pharma_val)
                        doses.append((date_obj, brand))
        # Sort doses chronologically
        doses.sort(key=lambda x: x[0])
        dose_data.append((person_id, doses))

    # Build dose DataFrame
    dose_rows = []
    for person_id, doses in dose_data:
        for dose_idx, (dose_date, dose_brand) in enumerate(doses, start=1):
            dose_rows.append(
                {
                    "ID": person_id,
                    "dose_idx": dose_idx,
                    "VDate": dose_date,
                    "VBrand": dose_brand,
                }
            )

    if dose_rows:
        dose_df = pd.DataFrame(dose_rows)
        # Pivot to wide format
        dates_wide = dose_df.pivot(index="ID", columns="dose_idx", values="VDate")
        brands_wide = dose_df.pivot(index="ID", columns="dose_idx", values="VBrand")
        dates_wide.columns = [f"V{c}Date" for c in dates_wide.columns]
        brands_wide.columns = [f"V{c}Brand" for c in brands_wide.columns]
        wide = pd.concat([dates_wide, brands_wide], axis=1)
    else:
        wide = pd.DataFrame()

    # Death date - create mapping from ID to death date
    death_series = pd.to_datetime(df.get("date_death", pd.Series([None] * len(df))), errors="coerce")
    death_df = pd.DataFrame({"id": df["id"], "death": death_series})
    death_per_id = death_df.groupby("id")["death"].min()

    # Assemble output
    # Get all unique IDs from original dataframe (include people with and without doses)
    all_ids = df["id"].drop_duplicates().values
    
    # Start with all IDs, then merge in dose data
    if not wide.empty:
        # Merge dose data for people who have doses
        out = pd.DataFrame({"ID": all_ids})
        wide_reset = wide.reset_index()
        out = out.merge(wide_reset, on="ID", how="left")
    else:
        # If no doses, create DataFrame with unique IDs only
        out = pd.DataFrame({"ID": all_ids})

    # Map per-ID fields - create mapping from ID to YOB
    id_to_yob = dict(zip(df["id"], yob_series))
    out["YearOfBirth"] = out["ID"].map(id_to_yob).astype("Int64").astype(str).replace({"<NA>": ""})

    out["DeathDate"] = out["ID"].map(death_per_id)
    out["Sex"] = ""  # No sex column in japan2 format
    out["DCCI"] = ""
    out["CensorDate"] = ""

    # Order columns
    base_cols = ["ID", "YearOfBirth", "DeathDate", "Sex", "DCCI", "CensorDate"]
    dose_cols = sorted(
        [c for c in out.columns if c.startswith("V")],
        key=lambda x: (int(re.sub(r"[^0-9]", "", x) or 0), x.endswith("Brand")),
    )
    out = out[base_cols + dose_cols]

    # Convert dates to ISO strings
    for c in [col for col in out.columns if col.endswith("Date")]:
        s = pd.to_datetime(out[c], errors="coerce")
        out[c] = s.dt.strftime("%Y-%m-%d").fillna("")

    # Restore original first-appearance ID order
    out["_order"] = out["ID"].map(first_order)
    out = out.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    
    # Print summary statistics
    unique_ids = out["ID"].nunique()
    ids_with_doses = out[[c for c in out.columns if c.startswith("V") and c.endswith("Date")]].notna().any(axis=1).sum()
    ids_with_death = out["DeathDate"].ne("").sum()
    print(f"Conversion complete:", flush=True)
    print(f"  Unique IDs: {unique_ids}", flush=True)
    print(f"  IDs with vaccination doses: {ids_with_doses}", flush=True)
    print(f"  IDs with death dates: {ids_with_death}", flush=True)
    
    return out


def write_output_csv(df: pd.DataFrame, output_path: str) -> None:
    """Write KRF CSV output."""
    df.to_csv(output_path, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)


def write_sidecar_yaml(
    sidecar_path: Optional[str],
    observation_end: Optional[str],
    birth_band_years: int,
    timezone: str,
) -> None:
    """Write sidecar YAML metadata."""
    if not sidecar_path:
        return
    data = {"schema": "KRF-1.0", "timezone": timezone, "birthBandYears": int(birth_band_years)}
    if observation_end:
        try:
            _ = dt.datetime.fromisoformat(observation_end).date()
            data["observationEndDate"] = observation_end
        except Exception:
            pass

    if yaml is not None:
        with open(sidecar_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    else:
        with open(sidecar_path, "w", encoding="utf-8") as f:
            for k, v in data.items():
                f.write(f"{k}: {v}\n")


def main() -> None:
    args = parse_args()

    def resolve_config_path(value: Optional[str], config_path: Optional[str]) -> Optional[str]:
        if not value or not config_path or os.path.isabs(value):
            return value
        config_dir = os.path.dirname(os.path.abspath(config_path))
        if value.startswith("data/") or value.startswith(f"data{os.sep}"):
            repo_root = os.path.abspath(os.path.join(config_dir, os.pardir, os.pardir))
            return os.path.normpath(os.path.join(repo_root, value))
        return os.path.normpath(os.path.join(config_dir, value))

    # If YAML config provided, override CLI values BEFORE reading input
    if args.config and os.path.exists(args.config) and yaml is not None:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        args.input = cfg.get("input", args.input)
        args.output = cfg.get("output", args.output)
        args.observation_end = cfg.get("observationEndDate", args.observation_end)
        args.birth_band_years = cfg.get("birthBandYears", args.birth_band_years)
        args.timezone = cfg.get("timezone", args.timezone)
        args.start_date = cfg.get("startDate", args.start_date)
        args.end_date = cfg.get("endDate", args.end_date)

    args.input = resolve_config_path(args.input, args.config)
    args.output = resolve_config_path(args.output, args.config)

    input_path = args.input
    output_path = args.output
    sidecar_path = args.sidecar

    if not input_path:
        raise SystemExit("Input path must be specified via --input or config YAML 'input'.")

    inputs = resolve_inputs(input_path)
    if not inputs:
        raise SystemExit(f"No input files found under: {input_path}")

    print(f"\n{'='*60}", flush=True)
    print(f"Starting conversion: {len(inputs)} file(s) -> {output_path}", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    df = read_concat(inputs)
    if df.empty:
        raise SystemExit(f"No data read from input files: {inputs}")

    out_df = build_output(df, args.start_date, args.end_date)
    
    print(f"\nWriting output to {output_path}...", flush=True)
    write_output_csv(out_df, output_path)
    output_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
    output_size_mb = output_size / (1024 * 1024)
    print(f"Wrote {len(out_df)} rows to {output_path} ({output_size_mb:.1f} MB)", flush=True)

    # Avoid overwriting the converter config YAML if sidecar path equals config path
    if sidecar_path and (not args.config or os.path.abspath(sidecar_path) != os.path.abspath(args.config)):
        write_sidecar_yaml(
            sidecar_path,
            observation_end=args.observation_end,
            birth_band_years=args.birth_band_years,
            timezone=args.timezone,
        )


if __name__ == "__main__":
    main()

