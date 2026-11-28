"""19.2-19.4/19.7: test the record converter and actual Makefile in scratch."""
from pathlib import Path
import csv
import os
import shutil
import subprocess
import sys
import tempfile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ENROLL = "2021_24"


def fixture(path, interior_gap=False):
    """Six selected people, plus one earlier death and one out-of-frame birth."""
    header = next(csv.reader((ROOT / "data/Czech/records_10k.csv").open(
        encoding="utf-8", newline="")))
    records = [
        (1940, 0, None), (1940, 1, "2021-27"),
        (1960, 0, "2021-28"), (1960, 1, None),
        (1970, 1, "2021-40"), (1970, 0, None),
        (1940, 0, "2021-20"), (1980, 0, "2022-10"),
    ]
    if interior_gap:
        # No exactly-once-vaccinated person; 3+ still pools at historical cap 2.
        records = [(y, 3 if d else 0, death) for y, d, death in records]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, (yob, dose, death) in enumerate(records):
            row = [""] * len(header)
            row[0], row[2], row[3], row[50], row[52] = i, 1, yob, death, 0
            for j in range(dose):
                row[10 + j] = f"2021-{10 + 3*j:02d}"
                row[17 + j] = "CO01"
            writer.writerow(row)
    return records


def run_converter(t, low=1930, high=1970, aggregates=1, interior_gap=False):
    src, dst = t / "records.csv", t / "mc.xlsx"
    records = fixture(src, interior_gap)
    env = dict(os.environ, MONTE_CARLO="1", MC_THREADS="1", MC_ITERATIONS="4",
               DATASET="Czech", MC_ENROLLMENT_DATE=ENROLL, MC_YOB_START=str(low),
               MC_YOB_END=str(high), MC_INCLUDE_AGGREGATES=str(aggregates),
               BYPASS_FREEZE="0", PYTHONIOENCODING="utf-8")
    result = subprocess.run([sys.executable, str(ROOT / "code/KCOR_CMR.py"),
                             str(src), str(dst)], env=env, capture_output=True,
                            text=True, encoding="utf-8", errors="replace")
    assert result.returncode == 0, result.stdout + result.stderr
    df = pd.read_excel(dst, sheet_name=ENROLL)
    master = pd.DataFrame(records, columns=["YearOfBirth", "Dose", "death"])
    master = master.loc[master.YearOfBirth.between(low, high)
                        & (master.death.isna() | master.death.ge("2021-24"))]
    master = master.reset_index(drop=True)
    master["Dose"] = master.Dose.clip(upper=2)
    assert set(df.mc_id) == set(range(4))
    assert df.MCSourceYOBStart.eq(low).all() and df.MCSourceYOBEnd.eq(high).all()
    assert df.MCDoseCap.eq(2).all() and df.MCPlannedDraws.eq(3).all()
    end = master.death.max(skipna=True)
    for i, draw in df.groupby("mc_id"):
        sampled = master if i == 0 else master.sample(n=len(master), replace=True, random_state=i)
        raw = draw[draw.YearOfBirth.ge(0)]
        assert set(raw.YearOfBirth) == set(sampled.YearOfBirth)
        assert raw.ISOweekDied.min() == "2021-24" and raw.ISOweekDied.max() == end
        assert set(raw.Dose) == {0, 1, 2}
        for (y, d), rows in raw.groupby(["YearOfBirth", "Dose"]):
            people = sampled[sampled.YearOfBirth.eq(y) & sampled.Dose.eq(d)]
            rows = rows.sort_values("ISOweekDied")
            assert rows.Alive.iloc[0] == len(people)
            for row in rows.itertuples():
                assert row.Dead == people.death.eq(row.ISOweekDied).sum()
                assert row.Alive == len(people) - people.death.lt(row.ISOweekDied).sum()
        zero_dose = 1 if interior_gap else 2
        assert raw.loc[raw.Dose.eq(zero_dose), ["Alive", "Dead"]].eq(0).all().all()
        if aggregates:
            for label, source in ((-2, raw), (-60, raw[raw.YearOfBirth.ge(1960)])):
                expected = source.groupby(["ISOweekDied", "Dose"])[["Alive", "Dead"]].sum()
                actual = draw[draw.YearOfBirth.eq(label)].set_index(["ISOweekDied", "Dose"])[["Alive", "Dead"]]
                pd.testing.assert_frame_equal(expected.sort_index(), actual.sort_index())
        else:
            assert draw.YearOfBirth.ge(0).all()
    return df


def converter_checks():
    with tempfile.TemporaryDirectory() as temp:
        t = Path(temp)
        broad = run_converter(t)
        point = broad[broad.mc_id.eq(0) & broad.YearOfBirth.ge(0)]
        assert point.loc[point.ISOweekDied.eq("2021-24"), "Alive"].sum() == 6
        assert point.Dead.sum() == 3
        assert point.loc[point.YearOfBirth.eq(1970), "Dead"].sum() == 1
        # Draw 2 omits the last-death person, yet retains the complete calendar.
        draw2 = broad[broad.mc_id.eq(2) & broad.YearOfBirth.eq(-2)]
        assert draw2.loc[draw2.ISOweekDied.eq("2021-40"), "Dead"].sum() == 0
        assert draw2.ISOweekDied.max() == "2021-40"
        default = run_converter(t, high=1960)
        point = default[default.mc_id.eq(0) & default.YearOfBirth.ge(0)]
        assert point.loc[point.ISOweekDied.eq("2021-24"), "Alive"].sum() == 4
        assert point.Dead.sum() == 2
        run_converter(t, aggregates=0, interior_gap=True)
        print("PASS actual converter: nondefault years, default scope, zero highest/interior doses, capped doses, shared calendar, exact sampled records and aggregates")


def make_checks():
    make = shutil.which("make")
    if not make:
        raise RuntimeError("GNU make is required for the configuration regression")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    shell = []
    if os.name == "nt":
        git_sh = Path("C:/Program Files/Git/usr/bin/sh.exe")
        if git_sh.exists():
            env["PATH"] = str(git_sh.parent) + os.pathsep + env["PATH"]
            shell = ["SHELL=" + git_sh.as_posix()]
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        code, data = root / "code", root / "data" / "Probe"
        code.mkdir()
        data.mkdir(parents=True)
        shutil.copy2(ROOT / "code/Makefile", code / "Makefile")
        # Probe the actual dependency graph and environment wiring without
        # running the expensive real estimator on every configuration change.
        for name, tag in (("KCOR_CMR.py", "producer"), ("KCOR.py", "consumer")):
            (code / name).write_text(
                "import os, pathlib, sys\n"
                "with open('calls.txt','a') as f:\n"
                f"    print('{tag}', os.environ['MC_YOB_END'], os.environ['MC_INCLUDE_AGGREGATES'], "
                "os.environ['MC_ENROLLMENT_DATE'], os.environ['MC_ITERATIONS'], file=f)\n"
                f"if os.environ.get('PROBE_FAIL') == '{tag}': sys.exit(9)\n"
                "pathlib.Path(sys.argv[2]).write_text('ok')\n",
                encoding="utf-8")
        (code / "krf_adapter.py").write_text("# scratch placeholder\n")
        (data / "Probe.yaml").write_text("enrollmentDates: [2021-24]\n")
        (data / "records.csv").write_text("scratch\n")
        calls = code / "calls.txt"
        stamp = data / ".kcor_mc_active.stamp"

        def run(expected, **settings):
            args = [make, *shell, "monte_carlo", "DATASET=Probe",
                    f"PYTHON={sys.executable.replace(os.sep, '/')}",
                    f"TMP_MC={(root / 'temporary.xlsx').as_posix()}"]
            args.extend(f"{k}={v}" for k, v in settings.items())
            proc = subprocess.run(args, cwd=code, env=env, capture_output=True,
                                  text=True, encoding="utf-8", errors="replace")
            assert proc.returncode == 0, proc.stdout + proc.stderr
            actual = calls.read_text().splitlines()
            assert len(actual) == 2 * expected, (actual, proc.stdout)
            assert actual[-2].replace("producer", "consumer") == actual[-1]
            return stamp.stat().st_mtime_ns

        first = run(1)
        assert run(1) == first
        run(2, MC_YOB_END=1970)
        run(3)
        run(4, MC_INCLUDE_AGGREGATES=0)
        run(5)
        run(6, MC_ENROLLMENT_DATE="2022_06")
        run(7)
        run(8, MC_ITERATIONS=6)
        final = run(9)
        assert run(9) == final
        assert calls.read_text().splitlines()[-2:] == [
            "producer 1960 1 2021_24 4", "consumer 1960 1 2021_24 4"]
        # A failed producer must not certify its old workbook for the changed
        # config, despite having updated the desired configuration stamp.
        args = [make, *shell, "monte_carlo", "DATASET=Probe",
                f"PYTHON={sys.executable.replace(os.sep, '/')}",
                f"TMP_MC={(root / 'temporary.xlsx').as_posix()}", "MC_YOB_END=1970"]
        failed = subprocess.run(args, cwd=code, env=dict(env, PROBE_FAIL="producer"),
                                capture_output=True, text=True, encoding="utf-8", errors="replace")
        assert failed.returncode != 0
        assert len(calls.read_text().splitlines()) == 19
        retried = subprocess.run(args, cwd=code, env=env, capture_output=True,
                                 text=True, encoding="utf-8", errors="replace")
        assert retried.returncode == 0, retried.stdout + retried.stderr
        assert calls.read_text().splitlines()[-3:] == [
            "producer 1970 1 2021_24 4", "producer 1970 1 2021_24 4",
            "consumer 1970 1 2021_24 4"]
        print("PASS actual Makefile: unchanged run is fresh; changed range, aggregates, enrollment and iteration count rebuild both stages; restoring every setting rebuilds; failed producer retries")


if __name__ == "__main__":
    converter_checks()
    make_checks()
