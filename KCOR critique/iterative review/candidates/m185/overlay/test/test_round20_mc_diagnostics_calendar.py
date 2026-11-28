"""Round 20.1/20.2: MC diagnostic delivery and validated calendar."""
from test_round19_mc_reader import case, section, run, select, K, pd, np
from bootstrap_reporting import flagged
import sys


def identified_case(draw, ages=(1940, 1950, 1960)):
    # Reproduce blind B's 40-week fixture; leave two other ages for isolation.
    dates = pd.date_range("2023-01-02", periods=40, freq="7D")
    frames = []
    for age in ages:
        for dose in (0, 1):
            risk = 100000
            a, d = [], []
            for i in range(len(dates)):
                a.append(risk)
                dead = round(risk * (.008 if dose == 1 and 8 <= i <= 14 else .002))
                d.append(dead)
                risk -= dead
            frames.append(pd.DataFrame(dict(mc_id=draw, YearOfBirth=age, Dose=dose,
                Sex="F", DateDied=dates, ISOweekDied=dates.strftime("%G-%V"), Alive=a, Dead=d)))
    return pd.concat(frames, ignore_index=True)


def quiet_window():
    K._load_dataset_dose_pairs_config()
    K._DATASET_THETA_ESTIMATION_WINDOWS_CACHE = [
        dict(start_iso="2023-14", end_iso="2023-35", start_int=202314,
             end_int=202335, label="Round 20 test")]


def endpoints(out, age=-2):
    return out[out.YearOfBirth.eq(age)].sort_values("Date").groupby("mc_id").tail(1).set_index("mc_id")


def numeric_intervals(out, total, audit_only):
    e = endpoints(out)
    lo, hi = np.quantile(e.loc[[1, 2, 3, 4], "KCOR"], [.025, .975])
    point = e.loc[0, "KCOR"]
    assert f"Percentile 95% CI: [{lo:.4f}, {hi:.4f}]" in total
    assert f"Basic bootstrap 95% CI: [{2*point-hi:.4f}, {2*point-lo:.4f}]" in total
    lines = [line for line in total.splitlines() if "95% CI:" in line]
    assert len(lines) == 2 and all(("audit-only" in line) == audit_only for line in lines)


def flags_and_intervals():
    # Blind A: finite fallback estimates remain in the complete draw roster.
    f = pd.concat([case(i, 20) for i in range(5)], ignore_index=True)
    out, _, log = run(f.assign(MCPlannedDraws=4), write=True, low=1940, high=1960)
    e = endpoints(out)
    assert list(e.index.sort_values()) == list(range(5))
    assert e.KCOR.notna().all()
    assert e.theta0_status_num.eq("NOT_IDENTIFIED").all() and e.theta0_status_den.eq("NOT_IDENTIFIED").all()
    assert e.normalization_status.eq("ok").all()
    assert flagged(e).all()
    total = section(log, "1 vs 0 - All Ages")
    assert total.count("fit_flagged (audit-only)") == 5
    assert "Point estimate (iteration 0):" in total and "(fit-flagged; audit-only)" in total
    assert "Bootstrap draws: 4 valid / 4 required" in total
    assert "Numerically complete draws with fit flags: 4; IDs [1, 2, 3, 4]" in total
    assert "Residual-screen flagged draws:" in total and "Residual-screen unavailable draws:" in total
    assert "Gap-assumption interval screen: not assessed" in total
    numeric_intervals(out, total, True)
    print("PASS: all five finite fallback rows flagged, four draws retained, both interval values preserved and audit-only")

    quiet_window()
    f = pd.concat([identified_case(i) for i in range(5)], ignore_index=True)
    out, _, log = run(f.assign(MCPlannedDraws=4), write=True, high=1960)
    e = endpoints(out)
    assert e.KCOR.notna().all() and not flagged(e).any(), e[["theta0_status_num", "theta0_status_den"]]
    total = section(log, "1 vs 0 - All Ages")
    assert "Numerically complete draws with fit flags: 0; IDs []" in total
    assert "fit_flagged (audit-only)" not in total and "fit-flagged; audit-only" not in total
    numeric_intervals(out, total, False)
    print("PASS: well-identified written-workbook counterexample has no fit-flag qualification")

    # An original-only flag and a draw-only flag each mark both intervals.
    # Inject only a fit status in the ordinary reporting predicate's input
    # via direct summary invocation to isolate this logic.
    for draw in (0, 2):
        probe = out.copy()
        probe.loc[probe.mc_id.eq(draw), "theta0_status_num"] = "NOT_IDENTIFIED"
        messages = []
        K.create_mc_summary(probe, messages.append, "2023_01", input_draws=range(5), planned_draw_counts=[4])
        total = section("\n".join(messages), "1 vs 0 - All Ages")
        assert total.count("fit_flagged (audit-only)") == 1
        assert ("(fit-flagged; audit-only)" in total) == (draw == 0)
        assert f"Numerically complete draws with fit flags: {int(draw != 0)}" in total
        numeric_intervals(probe, total, True)
    print("PASS: original-only and draw-only fit flags independently qualify unchanged intervals")


def valid_input():
    quiet_window()
    return pd.concat([identified_case(i) for i in range(5)], ignore_index=True).assign(MCPlannedDraws=4)


def calendar():
    f = valid_input()
    baseline, _, _ = run(f, write=True, high=1960)
    assert not flagged(endpoints(baseline)).any()
    # Equivalent separator/zero-padding variants within the same calendar,
    # including a boundary between weeks 09 and 10.
    equivalent = f.copy()
    different = equivalent.Sex.eq("F") & equivalent.Dose.eq(1)
    equivalent.loc[different, "ISOweekDied"] = equivalent.loc[different, "ISOweekDied"].str.replace("-", "_", regex=False).str.replace("_0", "_", regex=False)
    out, _, _ = run(equivalent, write=True, high=1960)
    pd.testing.assert_frame_equal(baseline, out, check_exact=True)
    print("PASS: equivalent ISO separators and unpadded week yield identical written workbook")

    bad = f.copy()
    # Shift labels for one source group, with every label still unique.
    affected = bad.mc_id.eq(1) & bad.YearOfBirth.eq(1940) & bad.Dose.eq(1)
    bad.loc[affected, "ISOweekDied"] = (bad.loc[affected, "DateDied"] + pd.Timedelta(weeks=8)).dt.strftime("%G-%V")
    # Duplicate a label across two actual calendar weeks of another source group.
    affected = bad.mc_id.eq(2) & bad.YearOfBirth.eq(1940) & bad.Dose.eq(1)
    indices = bad.index[affected]
    bad.loc[indices[10], "ISOweekDied"] = bad.loc[indices[9], "ISOweekDied"]
    # Malformed and missing labels are not accepted if the column is supplied.
    affected = bad.mc_id.eq(3) & bad.YearOfBirth.eq(1940) & bad.Dose.eq(1)
    indices = bad.index[affected]
    bad.loc[indices[10], "ISOweekDied"] = "2023-11-extra"
    bad.loc[indices[11], "ISOweekDied"] = None
    out, _, log = run(bad, write=True, high=1960)
    for draw in (1, 2, 3):
        for age in (1940, -2, 0):
            x = select(out, draw, age)
            assert len(x) == 40, (draw, age, len(x))
            assert x.normalization_status.eq("invalid_iso_calendar").all(), (draw, age, x.normalization_status.unique())
            assert x[["KCOR", "KCOR_unanchored", "KCOR_ns"]].isna().all().all()
        for age in (1950, 1960):
            pd.testing.assert_frame_equal(select(baseline, draw, age), select(out, draw, age), check_exact=True)
    for draw in (0, 4):
        pd.testing.assert_frame_equal(baseline[baseline.mc_id.eq(draw)], out[out.mc_id.eq(draw)], check_exact=True)
    total = section(log, "1 vs 0 - All Ages")
    assert "Unavailable draws (invalid_iso_calendar): 3; IDs [1, 2, 3]" in total
    assert "Bootstrap draws: 1 valid / 4 required" in total and "Interval unavailable" in total
    assert "95% CI:" not in total
    assert "discordant ISOweekDied and DateDied" in log
    print("PASS: shifted, duplicate, malformed and missing ISO labels cause explicit affected-group unavailability; unrelated groups and draws exactly unchanged")

    # An absent label column is not a contradictory supplied calendar.
    out, _, _ = run(f.drop(columns="ISOweekDied"), write=True, high=1960)
    pd.testing.assert_frame_equal(baseline, out, check_exact=True)
    print("PASS: omitted optional label is derived from validated dates")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        # Separate-process m184/m185 comparison on identical valid input.
        out, _, _ = run(valid_input(), write=True, high=1960)
        out.to_pickle(sys.argv[1])
        print("Valid calendar written workbook:", out.shape, endpoints(out).KCOR.to_dict())
    else:
        flags_and_intervals()
        calendar()

