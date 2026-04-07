#!/usr/bin/env python3
"""Regression: NPH post-inversion tail must not create a step at endDate."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def test_nph_tail_continuity() -> None:
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "code"))
    from KCOR import apply_nph_correction_post_inversion  # noqa: PLC0415

    n = 6
    t_vals = np.arange(n, dtype=float)
    t_reb = t_vals - 2.0
    inc = 0.001 + 0.0001 * np.arange(n, dtype=float)
    H0 = np.cumsum(inc)
    dates = np.array(
        [np.datetime64("2022-01-03") + np.timedelta64(7 * i, "D") for i in range(n)],
        dtype="datetime64[ns]",
    )
    start_dt = dates[1]
    end_dt = dates[2]

    H_corr, diag = apply_nph_correction_post_inversion(
        H0,
        t_vals,
        dates,
        0.05,
        1.5,
        start_dt,
        end_dt,
        1e-5,
        0.01,
        t_rebased=t_reb,
    )

    i_end = 2
    assert i_end < n - 1
    assert abs(H_corr[i_end + 1] - H_corr[i_end] - (H0[i_end + 1] - H0[i_end])) < 1e-9
    for j in range(i_end + 2, n):
        assert abs(H_corr[j] - H_corr[j - 1] - (H0[j] - H0[j - 1])) < 1e-9

    off = float(H_corr[i_end] - H0[i_end])
    for j in range(i_end + 1, n):
        assert abs(H_corr[j] - (H0[j] + off)) < 1e-9

    assert abs(float(diag["post_wave_level_shift"]) - off) < 1e-9


if __name__ == "__main__":
    test_nph_tail_continuity()
    print("PASS nph post-window continuity")
