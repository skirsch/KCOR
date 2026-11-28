"""Compare revised workbook outputs for the same Czech sample alone and in a batch."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path, help="Contains alone/ and combined/ CLI outputs.")
    root = parser.parse_args().directory
    keys = ["Date", "YearOfBirth", "Dose_num", "Dose_den"]
    def curves(folder):
        rows = pd.read_excel(root / folder / "output.xlsx", sheet_name="2021_24")
        return rows.dropna(subset=["mc_id", *keys])

    alone, batch = curves("alone"), curves("combined")
    a = alone.set_index(keys).sort_index()
    b = batch[batch.mc_id == 0].set_index(keys).sort_index()
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    print(f"Exactly equal original cohort-week/dose-pair rows: {len(a)}.")
    alone_log = (root / "alone/run.log").read_text(encoding="utf-8-sig")
    assert "Percentile 95% CI:" not in alone_log
    assert "Bootstrap draws: 0" in alone_log

    day = pd.Timestamp("2022-06-13")  # reporting Monday one year after 2021-06-14
    target = batch[(batch.Date == day) & (batch.mc_id > 0)]
    log = (root / "combined/run.log").read_text(encoding="utf-8-sig")
    stats = log.split("SUMMARY STATISTICS")[-1]
    for (year, num, den), group in target.groupby(keys[1:]):
        np.testing.assert_array_equal(np.sort(group.mc_id), np.arange(1, 25))
        low, high = group.KCOR.quantile([.025, .975])
        section = stats.split(f"Dose combination: {int(num)} vs {int(den)} - Year {int(year)}")[1]
        section = section.split("Dose combination:")[0]
        assert f"Percentile 95% CI: [{low:.4f}, {high:.4f}]" in section
        assert "Bootstrap draws: 24" in section
        print(f"{int(year)}, {int(num)} vs {int(den)}: [{low:.4f}, {high:.4f}]")
    assert target.groupby(keys[1:]).ngroups == 3


if __name__ == "__main__":
    main()
