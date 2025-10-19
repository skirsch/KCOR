import argparse
import csv
import datetime as dt
import glob
import os
from typing import List, Optional

import pandas as pd

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Furst dataset to KRF CSV (minimal mapping)")
    parser.add_argument("-c", "--config", default=None, help="Path to converter config YAML")
    parser.add_argument("-i", "--input", required=False, help="Path to input CSV/CSV.GZ or a directory to glob 'dataset*.csv'")
    parser.add_argument(
        "-o",
        "--output",
        default="furst_krf.csv",
        help="Path to output KRF CSV (default: furst_krf.csv)",
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
        default=1,
        choices=[1, 5, 10],
        help="Birth band width metadata for sidecar (default: 1)",
    )
    parser.add_argument(
        "--reference-date",
        default="2021-01-01",
        help="Reference date for YearOfBirth derivations; fixed to 2021-01-01 by default",
    )
    parser.add_argument(
        "--timezone",
        default="local",
        help="Timezone label to record in sidecar (default: local)",
    )
    return parser.parse_args()


def resolve_inputs(input_arg: str) -> List[str]:
    if not input_arg:
        return []
    if os.path.isdir(input_arg):
        # Prefer a canonical data.csv; fall back to dataset*.csv*
        preferred = os.path.join(input_arg, "data.csv")
        if os.path.exists(preferred):
            return [preferred]
        files = sorted(glob.glob(os.path.join(input_arg, "dataset*.csv*")))
        return files
    return [input_arg]


def read_concat(inputs: List[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in inputs:
        try:
            df = pd.read_csv(
                path,
                header=0,
                dtype=str,
                na_values=["", "NULL", "null", "NaN"],
                keep_default_na=False,
                encoding="utf-8",
                low_memory=False,
            )
        except Exception:
            # Fall back to headerless
            df = pd.read_csv(
                path,
                header=None,
                dtype=str,
                na_values=["", "NULL", "null", "NaN"],
                keep_default_na=False,
                encoding="utf-8",
                low_memory=False,
            )
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0, ignore_index=True)


def build_krf_from_days(df: pd.DataFrame, origin: dt.date) -> pd.DataFrame:
    # Expect two columns per README: [death_day, vax_day], days since origin; 0 means none
    if df.empty:
        return pd.DataFrame({
            "ID": ["1"],
            "YearOfBirth": ["1950"],
            "DeathDate": [""],
            "Sex": [""],
            "DCCI": [""],
            "CensorDate": [""],
        })

    # Coerce first two columns to integers (days), treating invalid/empty as 0
    first_two = df.iloc[:, :2].copy()
    for i in range(first_two.shape[1]):
        first_two.iloc[:, i] = pd.to_numeric(first_two.iloc[:, i], errors="coerce").fillna(0).astype(int)

    death_days = first_two.iloc[:, 0]
    vax_days = first_two.iloc[:, 1]

    def add_days(days: int) -> str:
        if days and days > 0:
            d = origin + dt.timedelta(days=int(days))
            return d.isoformat()
        return ""

    death_dates = death_days.apply(add_days)
    v1_dates = vax_days.apply(add_days)

    out = pd.DataFrame({
        "ID": [str(i + 1) for i in range(len(df))],
        "YearOfBirth": ["1950"] * len(df),
        "DeathDate": death_dates,
        "Sex": ["" ] * len(df),
        "DCCI": ["" ] * len(df),
        "CensorDate": ["" ] * len(df),
        "V1Date": v1_dates,
        # V1Brand intentionally omitted (unknown)
    })
    return out


def write_output_csv(df: pd.DataFrame, output_path: str) -> None:
    df.to_csv(output_path, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)


def write_sidecar_yaml(sidecar_path: Optional[str], observation_end: Optional[str], birth_band_years: int, timezone: str) -> None:
    if not sidecar_path:
        return
    data = {
        "schema": "KRF-1.0",
        "timezone": timezone,
        "birthBandYears": int(birth_band_years),
    }
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

    # Allow YAML config to override CLI
    if args.config and os.path.exists(args.config) and yaml is not None:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        args.input = cfg.get("input", args.input)
        args.output = cfg.get("output", args.output)
        args.observation_end = cfg.get("observationEndDate", args.observation_end)
        args.birth_band_years = cfg.get("birthBandYears", args.birth_band_years)
        args.timezone = cfg.get("timezone", args.timezone)
        # For Furst, we keep a fixed reference date; allow override
        args.reference_date = cfg.get("referenceDate", args.reference_date)

    input_path = args.input
    output_path = args.output
    sidecar_path = args.sidecar

    if not input_path:
        raise SystemExit("Input path must be specified via --input or config YAML 'input'.")

    inputs = resolve_inputs(input_path)
    if not inputs:
        raise SystemExit(f"No input files found under: {input_path}")

    raw = read_concat(inputs)
    try:
        origin_date = dt.datetime.fromisoformat(args.reference_date).date()
    except Exception:
        origin_date = dt.date(2021, 1, 1)
    out_df = build_krf_from_days(raw, origin_date)
    write_output_csv(out_df, output_path)

    if sidecar_path and (not args.config or os.path.abspath(sidecar_path) != os.path.abspath(args.config)):
        write_sidecar_yaml(
            sidecar_path,
            observation_end=args.observation_end,
            birth_band_years=args.birth_band_years,
            timezone=args.timezone,
        )


if __name__ == "__main__":
    main()



