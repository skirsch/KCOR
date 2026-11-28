"""Round 19.1, 19.6 and 19.7 reader regressions through Excel round trips."""
from test_round18_mc_sources import run, select, input_frame, pd, np, S, K
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import os, io, tempfile


def case(draw, weeks=53, years=(1940, 1950, 1960), zero_ref=False):
    dates = pd.date_range("2023-01-02", periods=weeks, freq="7D")
    frames = []
    for age in years:
        for dose in (0, 1):
            dead = np.arange(weeks) % 3 + 20 + (age-1940) + dose*4
            if dose:
                dead[12:] *= 2
                if zero_ref:
                    dead[:20] = 0
            alive = 20000 - np.r_[0, np.cumsum(dead[:-1])]
            frames.append(pd.DataFrame(dict(mc_id=draw, YearOfBirth=age, Dose=dose, Sex="F",
                ISOweekDied=dates.strftime("%G-%V"), DateDied=dates, Alive=alive, Dead=dead)))
    return pd.concat(frames, ignore_index=True)


def section(log, name):
    return log.split("Dose combination: " + name, 1)[1].split("Dose combination:", 1)[0]


def summary_dates_and_roster():
    f = pd.concat([case(0), case(1), case(2, 17), case(3, zero_ref=True),
                   case(4, years=(1970,)).assign(Dose=9)], ignore_index=True)
    f["MCPlannedDraws"] = 5
    out, _, log = run(f, write=True, low=1940, high=1960)
    total = section(log, "1 vs 0 - All Ages")
    individual = section(log, "1 vs 0 - Year 1940")
    print(total)
    assert "Requested reporting date: 2024-01-01" in log
    assert "Actual reporting date: 2024-01-01" in total
    assert "missing_reporting_week" in total and "undefined_reference_ratio" in total
    assert "missing_input_iteration" in total and "missing_comparison" in total
    assert "missing_comparison" in individual
    assert "Interval unavailable" in total and "95% CI:" not in total
    assert select(out, 3).KCOR_ns.isna().all()
    assert "Bootstrap draws: 1 valid / 5 required" in individual and "Bootstrap draws: 1 valid / 5 required" in total
    assert pd.to_datetime(select(out, 2).Date).max() == pd.Timestamp("2023-04-24")
    assert pd.to_datetime(select(out, 0).Date).max() == pd.Timestamp("2024-01-01")
    # Complete roster, both matching the original endpoint, preserves interval arithmetic.
    f = pd.concat([case(0, 20), case(1), case(2, 20)], ignore_index=True)
    f["MCPlannedDraws"] = 2
    out, _, log = run(f, write=True, low=1940, high=1960)
    total = section(log, "1 vs 0 - All Ages")
    assert "Actual reporting date: 2023-05-15" in total
    assert "Bootstrap draws: 2 valid / 2 required" in total
    date = pd.Timestamp("2023-05-15")
    endpoints = out[out.YearOfBirth.eq(-2) & pd.to_datetime(out.Date).eq(date)].set_index("mc_id").KCOR
    lo, hi = np.quantile(endpoints.loc[[1, 2]], [.025, .975])
    assert f"Percentile 95% CI: [{lo:.4f}, {hi:.4f}]" in total
    assert f"Basic bootstrap 95% CI: [{2*endpoints[0]-hi:.4f}, {2*endpoints[0]-lo:.4f}]" in total
    # Partial source signatures with no matching original must not print even a percentile CI.
    f = pd.concat([case(0, 20), case(1, 20, years=(1940, 1950))], ignore_index=True)
    _, _, log = run(f, write=True, scope="partial", high=1960)
    group1 = section(log, "1 vs 0 - All available ages (dose 1: 1940,1950,1960; dose 0: 1940,1950,1960)")
    group2 = section(log, "1 vs 0 - All available ages (dose 1: 1940,1950; dose 0: 1940,1950)")
    assert "different_population" in group1 and "Interval unavailable" in group1
    assert "no original week" in group2 and "Interval unavailable" in group2
    assert "95% CI:" not in group1 + group2
    print("PASS: exact original endpoint, complete roster, missing comparison/week/anchor/iteration, changed partial signature, and matched-week interval arithmetic")


def provenance():
    f = pd.concat([case(0, 20), case(1, 20)], ignore_index=True)
    f = f.assign(MCSourceYOBStart=1930, MCSourceYOBEnd=1960, MCDoseCap=1, MCPlannedDraws=1)
    out, _, log = run(f, write=True, high=1960)
    assert out.MCSourceYOBStart.eq(1930).all() and out.MCSourceYOBEnd.eq(1960).all()
    assert out.MCDoseCap.eq(1).all() and out.MCPlannedDraws.eq(1).all()
    assert out.Dose_num_label.eq("1+").all() and out.Dose_den_label.astype(str).eq("0").all()
    assert select(out, 0).population_label.eq("All Ages (source birth years 1930-1960)").all()
    # If an explicit -60 source is supplied, label its actual intersection.
    post = f[f.YearOfBirth.eq(1960)].assign(YearOfBirth=-60)
    out, _, log = run(pd.concat([f, post], ignore_index=True), write=True, high=1960)
    assert select(out, 0, -60).population_label.eq("-60 (born 1960+) (source birth years 1960)").all()
    assert "Dose combination: 1+ vs 0 - All Ages (source birth years 1930-1960)" in log
    assert "Dose 1+ includes all doses at or above 1" in log
    # A draw with differing selection or cap must not enter the original interval.
    shifted = f.copy()
    shifted.loc[shifted.mc_id.eq(1), "MCSourceYOBStart"] = 1940
    _, _, log = run(shifted, write=True, high=1960)
    total = section(log, "1+ vs 0 - All Ages (source birth years 1930-1960)")
    assert "different_population" in total and "95% CI:" not in total
    print("PASS: producer source range and cap in weekly workbook and summary; mismatched metadata withheld")


def raw_fractional():
    # Both the tiny positive-reference and tiny positive-denominator cases.
    for tiny_dose in (0, 1):
        frame = input_frame(S).copy()
        frame["Dead"] = frame.Dead.astype(float)
        frame["Alive"] = frame.Alive.astype(float)
        for _, cell in frame.groupby(["YearOfBirth", "Dose"]):
            if cell.Dose.iloc[0] == tiny_dose:
                deaths = np.full(len(cell), 1e-13)
                frame.loc[cell.index, "Dead"] = deaths
                frame.loc[cell.index, "Alive"] = 20000 - np.r_[0., np.cumsum(deaths[:-1])]
        mc, _, _ = run(frame.assign(mc_id=0), write=True)
        for age in (1940, -2):
            point = select(mc, 0, age)
            reference = point.loc[point.t.eq(K.KCOR_NORMALIZATION_WEEKS_EFFECTIVE)]
            assert len(reference) == 1 and reference.KCOR_ns.iloc[0] == 1.
        # Ordinary actual workbook path, not MC aggregation.
        os.environ.update(MONTE_CARLO="0", KCOR_BOOTSTRAP_DRAWS="0", KCOR_BOOTSTRAP_LAYOUT="disjoint")
        K.MONTE_CARLO_MODE=False
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            src=Path(tmp)/"in.xlsx"; dst=Path(tmp)/"out.xlsx"
            frame.to_excel(src,sheet_name=S,index=False)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                ordinary=K.process_workbook(str(src),str(dst),return_rows=True)
            # Ordinary writer does not expose this internal comparator on dose_pairs.
        for age in (1940,-2):
            reference = ordinary[ordinary.YearOfBirth.eq(age) & ordinary.t.eq(K.KCOR_NORMALIZATION_WEEKS_EFFECTIVE)]
            assert len(reference) == 1 and reference.KCOR_ns.iloc[0] == 1.
    print("PASS: finite tiny denominator and reference ratio, ordinary and MC individual and aggregate workbook paths")


if __name__ == "__main__":
    summary_dates_and_roster()
    provenance()
    raw_fractional()


