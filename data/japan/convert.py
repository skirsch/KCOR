import argparse
import csv
import datetime as dt
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


EXPECTED_HEADERS = [
    "ID",
    "last_dose",
    "age",
    "sex",
    "dose",
    "lot",
    "mfg",
    "vdate",
    "last_dose_died",
    "age_died",
    "unused",
    "DoD",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Japan vaccination CSV to KRF CSV")
    parser.add_argument("-c", "--config", default=None, help="Path to converter config YAML")
    parser.add_argument("-i", "--input", required=False, help="Path to input CSV/CSV.GZ")
    parser.add_argument(
        "-o",
        "--output",
        default="japan_krf.csv",
        help="Path to output KRF CSV (default: japan_krf.csv)",
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
        "--reference-date",
        default="earliest_vdate",
        help="Reference date for YearOfBirth: 'earliest_vdate' or explicit YYYY-MM-DD (default: earliest_vdate)",
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


def load_headers_fallback(col_headings_path: str) -> List[str]:
    if os.path.exists(col_headings_path):
        with open(col_headings_path, "r", encoding="utf-8") as f:
            line = f.readline().strip()
            if line:
                headers = [h.strip() for h in line.split(",")]
                return headers
    return EXPECTED_HEADERS.copy()


def try_read_with_header(input_path: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(
            input_path,
            dtype=str,
            na_values=["", "NULL", "null", "NaN"],
            keep_default_na=False,
            encoding="utf-8",
            low_memory=False,
        )
        # Consider successful only if at least the expected columns are present
        expected_set = set(EXPECTED_HEADERS)
        if expected_set.issubset(set(df.columns)):
            return df
        return None
    except Exception:
        return None


def read_input(input_path: str, col_headings_path: str) -> pd.DataFrame:
    df = try_read_with_header(input_path)
    if df is not None:
        return df

    headers = load_headers_fallback(col_headings_path)
    usecols = list(range(len(headers)))
    df = pd.read_csv(
        input_path,
        header=None,
        names=headers,
        usecols=usecols,
        dtype=str,
        na_values=["", "NULL", "null", "NaN"],
        keep_default_na=False,
        encoding="utf-8",
        low_memory=False,
    )
    return df


def normalize_sex(value: Optional[str]) -> str:
    if not value:
        return ""
    v = str(value).strip()
    if v in ("男", "M", "m", "male", "Male"):
        return "M"
    if v in ("女", "F", "f", "female", "Female"):
        return "F"
    return "O" if v else ""


def normalize_brand(value: Optional[str]) -> str:
    if not value:
        return ""
    v = str(value).strip()
    mapping = {
        "ファイザー": "P",
        "ﾌｧｲｻﾞｰ": "P",
        "Pfizer": "P",
        "ﾓﾃﾞﾙﾅ": "M",
        "モデルナ": "M",
        "Moderna": "M",
        "第一三共": "O",
        "アストラゼネカ": "O",
        "武田": "O",
        "ノババックス": "O",
        "AstraZeneca": "O",
        "Novavax": "O",
        "Other": "O",
    }
    return mapping.get(v, "O")


def parse_date(value: Optional[str]) -> str:
    if not value:
        return ""
    v = str(value).strip()
    if not v or v.upper() == "NULL":
        return ""
    try:
        # Expecting ISO, but be defensive
        dt_obj = dt.datetime.fromisoformat(v)
        return dt_obj.date().isoformat()
    except Exception:
        try:
            dt_obj = dt.datetime.strptime(v, "%Y/%m/%d")
            return dt_obj.date().isoformat()
        except Exception:
            return ""


def extract_age_band(age_str: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    if not age_str:
        return None, None
    s = str(age_str).strip()
    # Examples: "60～64歳", "100歳～"
    try:
        if "～" in s and s.endswith("歳"):
            left, right = s.split("～", 1)
            left = left.replace("歳", "").strip()
            right = right.replace("歳", "").strip()
            low = int(left)
            high = int(right)
            return low, high
        if s.endswith("歳～"):
            low = int(s.replace("歳～", "").strip())
            # Cap upper bound at 130 for open-ended upper age
            return low, 130
        # Fallback: single number with "歳"
        if s.endswith("歳"):
            n = int(s.replace("歳", "").strip())
            return n, n
    except Exception:
        return None, None
    return None, None


def pick_reference_date(person_df: pd.DataFrame, reference_date_arg: str) -> Optional[dt.date]:
    if reference_date_arg and reference_date_arg != "earliest_vdate":
        try:
            return dt.datetime.fromisoformat(reference_date_arg).date()
        except Exception:
            pass
    vdates = [parse_date(d) for d in person_df.get("vdate", [])]
    vdates = [d for d in vdates if d]
    if not vdates:
        return None
    return min(dt.datetime.fromisoformat(d).date() for d in vdates)


def compute_year_of_birth(age_band: Optional[str], reference_date: Optional[dt.date]) -> Optional[int]:
    if not age_band or not reference_date:
        return None
    low, high = extract_age_band(age_band)
    if high is None:
        return None
    # Clamp open-ended to 100 as requested; otherwise use given upper bound
    if high >= 130:
        high = 100
    # Use upper bound of age band so YearOfBirth encodes the lower bound of the birth-year band
    # YearOfBirth = reference_year - high
    try:
        return reference_date.year - int(high)
    except Exception:
        return None


def map_death_date(row: pd.Series) -> str:
    dod = parse_date(row.get("DoD"))
    if dod:
        return dod
    return parse_date(row.get("last_dose_died"))


def build_person_record(group: pd.DataFrame, reference_date_arg: str) -> Dict[str, str]:
    # This function is retained for compatibility but not used in the vectorized pipeline.
    # It can be used for small samples or unit tests.
    person_id = str(group["ID"].iloc[0]).strip() if "ID" in group.columns else ""
    sex_values = [normalize_sex(s) for s in group.get("sex", []) if str(s).strip() not in ("", "NULL")]
    sex_out = sex_values[0] if sex_values else ""

    ref_date = pick_reference_date(group, reference_date_arg)
    age_band = next((str(val) for val in group.get("age", []) if str(val).strip() and str(val).strip().upper() != "NULL"), None)
    yob = compute_year_of_birth(age_band, ref_date)
    yob_str = str(yob) if yob is not None else ""

    death_dates = [map_death_date(r) for _, r in group.iterrows()]
    death_dates = [d for d in death_dates if d]
    death_date = min(death_dates) if death_dates else ""

    # Doses
    doses = sorted({(parse_date(r.get("vdate")), normalize_brand(r.get("mfg"))) for _, r in group.iterrows() if parse_date(r.get("vdate"))})
    record: Dict[str, str] = {"ID": person_id, "YearOfBirth": yob_str, "DeathDate": death_date, "Sex": sex_out, "DCCI": "", "CensorDate": ""}
    for idx, (d, b) in enumerate(doses, start=1):
        record[f"V{idx}Date"] = d
        record[f"V{idx}Brand"] = b
    return record


def build_output(df: pd.DataFrame, reference_date_arg: str, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    df = df.fillna("")

    # Preserve first-appearance order of each ID (to match legacy output ordering)
    first_order = df.reset_index().groupby("ID")["index"].min()

    # Normalize columns we need
    df = df[[c for c in ["ID", "sex", "age", "vdate", "mfg", "DoD", "last_dose_died"] if c in df.columns]].copy()

    # Parse dates vectorized
    vdate = pd.to_datetime(df.get("vdate", pd.Series([None]*len(df))), errors="coerce")
    dod = pd.to_datetime(df.get("DoD", pd.Series([None]*len(df))), errors="coerce")
    ldd = pd.to_datetime(df.get("last_dose_died", pd.Series([None]*len(df))), errors="coerce")

    # Filter dose rows by window
    if start_date:
        sd = pd.to_datetime(start_date, errors="coerce")
        vdate = vdate.where((vdate.isna()) | (vdate >= sd))
    if end_date:
        ed = pd.to_datetime(end_date, errors="coerce")
        vdate = vdate.where((vdate.isna()) | (vdate <= ed))
    df["vdate_parsed"] = vdate

    # Sex mapping (trim and take first non-empty per ID)
    sex_map = {"男": "M", "女": "F", "M": "M", "F": "F", "male": "M", "female": "F", "Male": "M", "Female": "F"}
    if "sex" in df.columns:
        sex_series = df["sex"].astype(str).str.strip()
        sex_mapped = sex_series.map(sex_map)
        sex_nonempty = sex_mapped[sex_mapped.notna() & (sex_mapped != "")]
        sex_by_id = df.loc[sex_nonempty.index, ["ID"]].assign(Sex=sex_nonempty.values)
        sex_by_id = sex_by_id.drop_duplicates("ID").set_index("ID")["Sex"]
    else:
        sex_by_id = pd.Series(dtype=str)

    # Brand mapping
    brand_map = {"ファイザー": "P", "ﾌｧｲｻﾞｰ": "P", "Pfizer": "P", "ﾓﾃﾞﾙﾅ": "M", "モデルナ": "M", "Moderna": "M",
                 "第一三共": "O", "アストラゼネカ": "O", "武田": "O", "ノババックス": "O", "AstraZeneca": "O", "Novavax": "O", "Other": "O"}
    df["Brand"] = df.get("mfg", "").map(brand_map).fillna("O")

    # Earliest death date per ID
    death_series = pd.concat([dod, ldd], axis=1).min(axis=1)

    # Deduplicate same-day doses and sort
    dose_df = df[["ID", "vdate_parsed", "Brand"]].dropna(subset=["vdate_parsed"]).drop_duplicates(subset=["ID", "vdate_parsed"]).sort_values(["ID", "vdate_parsed"])  # type: ignore
    dose_df["dose_idx"] = dose_df.groupby("ID").cumcount() + 1

    # Pivot to wide
    dates_wide = dose_df.pivot(index="ID", columns="dose_idx", values="vdate_parsed")
    brands_wide = dose_df.pivot(index="ID", columns="dose_idx", values="Brand")
    # Flatten columns
    dates_wide.columns = [f"V{c}Date" for c in dates_wide.columns]
    brands_wide.columns = [f"V{c}Brand" for c in brands_wide.columns]
    wide = pd.concat([dates_wide, brands_wide], axis=1)

    # Reference date per ID
    if reference_date_arg == "earliest_vdate":
        ref_date_per_id = dose_df.groupby("ID")["vdate_parsed"].min()
    else:
        ref_fixed = pd.to_datetime(reference_date_arg, errors="coerce")
        ref_date_per_id = pd.Series(ref_fixed, index=wide.index)

    # YOB from age band
    if "age" in df.columns:
        age_clean = df["age"].astype(str).str.strip()
        age_clean = age_clean.where(~age_clean.str.upper().eq("NULL"), "")
        age_nonempty = df.loc[age_clean.ne("").values, ["ID"]].assign(age=age_clean[age_clean.ne("")].values)
        age_series = age_nonempty.drop_duplicates("ID").set_index("ID")["age"]
    else:
        age_series = pd.Series(dtype=str)
    # Extract upper bound and clamp 100+
    import re
    def age_high(s: str) -> Optional[int]:
        if not isinstance(s, str) or not s:
            return None
        s = s.strip()
        m = re.match(r"^(\d+)～(\d+)歳$", s)
        if m:
            return int(m.group(2))
        m = re.match(r"^(\d+)歳～$", s)
        if m:
            return 100
        m = re.match(r"^(\d+)歳$", s)
        if m:
            return int(m.group(1))
        return None
    age_high_series = age_series.map(age_high)
    yob = None
    if not ref_date_per_id.empty and not age_high_series.empty:
        ref_year = ref_date_per_id.dt.year
        # align indices
        ah = age_high_series.reindex(ref_year.index)
        yob = (ref_year - ah).astype("Int64")

    # Death per ID (min non-null)
    tmp = pd.DataFrame({"ID": df["ID"], "death": death_series})
    death_per_id = tmp.groupby("ID")["death"].min()

    # Assemble output
    out = wide.reset_index()
    out["YearOfBirth"] = yob.reindex(out.index).astype("Int64").astype(str).replace({"<NA>": ""}) if yob is not None else ""
    out["DeathDate"] = death_per_id.reindex(out.index)
    out["Sex"] = sex_by_id.reindex(out.index).fillna("")
    out["DCCI"] = ""
    out["CensorDate"] = ""

    # Order columns
    base_cols = ["ID", "YearOfBirth", "DeathDate", "Sex", "DCCI", "CensorDate"]
    dose_cols = sorted([c for c in out.columns if c.startswith("V")], key=lambda x: (int(re.sub(r"[^0-9]", "", x) or 0), x.endswith("Brand")))
    out = out[base_cols + dose_cols]

    # Convert dates to ISO strings
    for c in [col for col in out.columns if col.endswith("Date")]:
        s = pd.to_datetime(out[c], errors="coerce")
        out[c] = s.dt.strftime("%Y-%m-%d").fillna("")

    # Restore original first-appearance ID order
    out["_order"] = out["ID"].map(first_order)
    out = out.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    return out


def write_output_csv(df: pd.DataFrame, output_path: str) -> None:
    df.to_csv(output_path, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)


def write_sidecar_yaml(sidecar_path: str, observation_end: Optional[str], birth_band_years: int, timezone: str) -> None:
    if sidecar_path is None:
        return
    data = {
        "schema": "KRF-1.0",
        "timezone": timezone,
    }
    if observation_end:
        # validate format
        try:
            _ = dt.datetime.fromisoformat(observation_end).date()
            data["observationEndDate"] = observation_end
        except Exception:
            pass
    data["birthBandYears"] = int(birth_band_years)

    # Write YAML only if PyYAML is available
    if yaml is not None:
        with open(sidecar_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    else:
        # Fallback to a very simple YAML-like writer
        with open(sidecar_path, "w", encoding="utf-8") as f:
            for k, v in data.items():
                f.write(f"{k}: {v}\n")


def main() -> None:
    args = parse_args()

    # If YAML config provided, override CLI values BEFORE reading input
    if args.config and os.path.exists(args.config) and yaml is not None:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        args.input = cfg.get("input", args.input)
        args.output = cfg.get("output", args.output)
        args.observation_end = cfg.get("observationEndDate", args.observation_end)
        args.birth_band_years = cfg.get("birthBandYears", args.birth_band_years)
        args.timezone = cfg.get("timezone", args.timezone)
        args.reference_date = cfg.get("referenceDate", args.reference_date)
        args.start_date = cfg.get("startDate", args.start_date)
        args.end_date = cfg.get("endDate", args.end_date)

    input_path = args.input
    output_path = args.output
    sidecar_path = args.sidecar

    if not input_path:
        raise SystemExit("Input file must be specified via --input or config YAML 'input'.")

    col_headings_path = os.path.join(os.path.dirname(__file__), "col_headings.csv")
    df = read_input(input_path, col_headings_path)

    # Keep only expected columns if extras were present
    keep_cols = [c for c in EXPECTED_HEADERS if c in df.columns]
    df = df[keep_cols]

    out_df = build_output(df, args.reference_date, args.start_date, args.end_date)
    write_output_csv(out_df, output_path)

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


