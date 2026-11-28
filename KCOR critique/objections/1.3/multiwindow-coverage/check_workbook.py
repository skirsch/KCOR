"""Verify pooled fast-replay endpoints and reported percentiles against actual MC workbooks."""
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import study


def configure(scenario, fitter, stress=False):
    if stress:
        from fallback_stress import initialize
        initialize()
    else:
        study.initialize()
    model = study.model
    model._DATASET_DOSE_PAIRS_CACHE = {"2021_24": [(1, 0)], "default": [(1, 0)]}
    windows = []
    for a, b in study.WINDOWS:
        start, end = [study.DATES[i].strftime("%G-%V") for i in (a,b)]
        windows.append(dict(start_iso=start, end_iso=end,
                            start_int=model.iso_label_to_int(start), end_int=model.iso_label_to_int(end)))
    model._DATASET_THETA_ESTIMATION_WINDOWS_CACHE = windows
    if scenario == "nonlinear_wave":
        start, end = [study.DATES[i].strftime("%G-%V") for i in study.WAVE]
        model._DATASET_COVID_CORRECTION_CACHE = dict(startDate=start, endDate=end,
            mode="forced_alpha", forced_alpha=1.18, neutralization="symmetric_all_cohorts")
    else:
        model._DATASET_COVID_CORRECTION_CACHE = {}
    model.fit_theta0_gompertz = study.compare.joint_fit if fitter == "joint" else study.compare.production_fit
    return model


def write_workbook(source, out, scenario, fitter, stress=False):
    model = configure(scenario, fitter, stress)
    model.process_workbook(str(source), str(out))


def frame(batch):
    dead, alive = study.arrays(batch)
    return pd.concat([pd.DataFrame(dict(ISOweekDied=study.DATES.strftime("%G-%V"),
        DateDied=study.DATES, YearOfBirth=yob, Sex="O", Dose=dose,
        Alive=alive[draw, age, dose], Dead=dead[draw, age, dose], mc_id=draw))
        for draw in range(len(batch)) for age,yob in enumerate(study.STRATA) for dose in range(2)],
        ignore_index=True)


def main(stress=False):
    configure("nonlinear_wave", "fixed", stress)
    scenarios = ("nonlinear_wave",) if stress else study.SCENARIOS
    with tempfile.TemporaryDirectory(prefix="kcor-pooled-coverage-") as folder:
        root = Path(folder)
        for scenario in scenarios:
            rep = 0
            if stress:
                for rep in range(50):
                    sample, _ = study.samples(scenario, rep, 0)
                    outcomes = np.asarray(study.estimate(sample[0], study.pooling_weights(sample), scenario))
                    if np.any(outcomes[:,3] > 0):
                        break
                else:
                    raise ValueError("Could not find a flagged original sample for the stress workbook check.")
            batch, truth = study.samples(scenario, rep, 5)
            w = study.pooling_weights(batch)
            expected = np.asarray([study.estimate(c, w, scenario) for c in batch])
            source = root / f"{scenario}.xlsx"
            frame(batch).to_excel(source, sheet_name="2021_24", index=False)
            for j, fitter in enumerate(("fixed","joint")):
                output = root / f"{scenario}-{fitter}"
                output.mkdir()
                command = [sys.executable, "-B", str(Path(__file__).resolve()),
                           str(source), str(output/"output.xlsx"), scenario, fitter]
                if stress:
                    command.append("--stress")
                result = subprocess.run(command,
                                        cwd=output, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(result.stdout.decode(errors="replace")[-6000:])
                workbook = pd.read_excel(output/"output.xlsx", sheet_name="2021_24")
                final = workbook[(workbook.Date == study.DATES[-1]) & (workbook.Dose_num==1) & (workbook.Dose_den==0)]
                if final.empty or 0 not in final.YearOfBirth.values:
                    print("Reporting date", study.DATES[-1], "available last date", workbook.Date.max(),
                          "ages", sorted(workbook.YearOfBirth.unique()),
                          "reporting rows", final[["YearOfBirth","mc_id","KCOR"]].to_string(index=False))
                    print(result.stdout.decode(errors="replace")[-4000:])
                for col,yob in enumerate((0,*study.STRATA)):
                    rows = final[final.YearOfBirth==yob].sort_values("mc_id")
                    np.testing.assert_array_equal(rows.mc_id, np.arange(len(batch)))
                    np.testing.assert_allclose(rows.KCOR, expected[:,j,col], rtol=1e-10, atol=1e-12)
                log = result.stdout.decode(errors="replace").split("SUMMARY STATISTICS")[-1]
                # The summary suppresses age 0. Verify the actual pooled rows
                # and each emitted age-specific Monte Carlo percentile interval.
                for col,yob in enumerate(study.STRATA, 1):
                    low, high = np.quantile(expected[1:,j,col], [.025,.975])
                    assert f"Percentile 95% CI: [{low:.4f}, {high:.4f}]" in log, log[-6000:]
                print(f"{scenario} / {fitter}: pooled target, two strata and reported intervals matched.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) in (5,6):
        write_workbook(*sys.argv[1:5], stress="--stress" in sys.argv)
    else:
        main(stress="--stress" in sys.argv)
