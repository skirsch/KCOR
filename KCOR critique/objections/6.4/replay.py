from pathlib import Path
import os,sys,hashlib,json,tempfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "data/Czech/KCOR_CMR.xlsx"
os.environ.update(DATASET="Czech", KCOR_NPH_MODE="identified_only",
                  KCOR_BOOTSTRAP_DRAWS="0", SA_DOSE_PAIRS="2,0", MONTE_CARLO="0")
sys.path.insert(0, str(ROOT / "code"))
import KCOR as K

def main():
    with SOURCE.open("rb") as f:
        checksum = hashlib.file_digest(f, "sha256").hexdigest()
    columns = ["DateDied", "ISOweekDied", "YearOfBirth", "Dose", "Alive", "Dead"]
    raw = pd.read_excel(SOURCE, sheet_name="2021_24", usecols=columns)
    week = raw.ISOweekDied.astype(str).str.replace("-", "", regex=False).str[:6].astype(int)
    raw = raw[week.between(202124, 202416) & raw.Dose.isin([0, 2])].copy()
    # Age in calendar year 2024, grouped using the supplied five-year birth bands.
    known = raw[raw.YearOfBirth.gt(0)].copy()
    known["YearOfBirth"] = (known.YearOfBirth - 5) // 10 * 10 + 5
    known = known[known.YearOfBirth.between(1925, 1975)]
    supplied_total = raw[raw.YearOfBirth.eq(-2)]
    keys = ["YearOfBirth", "Dose", "DateDied", "ISOweekDied"]
    frame = pd.concat([known, supplied_total]).groupby(keys, as_index=False)[["Alive", "Dead"]].sum()
    frame["Sex"] = "O"
    K.AGE_RANGE = 5
    K.YEAR_RANGE = None
    K.OVERRIDE_DOSE_PAIRS = [(2, 0)]
    saved = {}
    original = K.build_kcor_rows

    def capture(df, sheet, *args, **kwargs):
        saved["frame"] = df.copy()
        params = args[2] if len(args) > 2 else kwargs["kcor6_params_map"]
        saved["params"] = params.copy()
        return original(df, sheet, *args, **kwargs)

    K.build_kcor_rows = capture
    try:
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            frame.to_excel(p / "input.xlsx", sheet_name="2021_24", index=False)
            K.process_workbook(str(p / "input.xlsx"), str(p / "output.xlsx"),
                               str(p / "run.log"), return_rows=True)
    finally:
        K.build_kcor_rows = original

    rows = []
    for (yob, dose), group in saved["frame"].groupby(["YearOfBirth", "Dose"]):
        if yob not in [-2, 1925, 1935, 1945, 1955, 1965, 1975]:
            continue
        group = group.sort_values("DateDied")
        assert len(group) == 149
        params = saved["params"][("2021_24", int(yob), int(dose))]
        hazard = K.hazard_from_mr(group.Dead.to_numpy(float) / group.Alive.to_numpy(float))
        effective = group.t.to_numpy() >= K.DYNAMIC_HVE_SKIP_WEEKS
        sampling = params.get("sampling_diagnostic", {})
        rows.append(dict(birth_band=int(yob), dose=int(dose), weeks=len(group),
                         raw_hazard_full=float(hazard.sum()), raw_hazard_after_skip=float(hazard[effective].sum()),
                         k=params["k_hat"], theta=params["theta_hat"], theta_raw=params.get("theta0_raw"),
                         status=params["theta0_status"], sampling_flag=sampling.get("sampling_flag"),
                         sampling_min_tail=sampling.get("sampling_min_tail"),
                         sampling_n=sampling.get("sampling_n"),
                         shape_windows=params.get("shape_windows", [])))
    assert len(rows) == 14
    result = dict(input=str(SOURCE.relative_to(ROOT)), sha256=checksum, sheet="2021_24",
                  endpoint="2024-16", mode="identified_only", skip_weeks=K.DYNAMIC_HVE_SKIP_WEEKS,
                  age_reference_year=2024, rows=rows)
    (HERE / "results.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(pd.DataFrame(rows).drop(columns="shape_windows").to_string(index=False), flush=True)

if __name__ == "__main__":
    main()
