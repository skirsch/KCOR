# Table 9 positive-control check

**Both published week-80 values can be reproduced from the raw workbook to all three reported decimal places. The supplied fitter command stops before fitting either scenario.** The frozen material supports an unadjusted positive-control comparison. It leaves the source of the historical table entries unverified.

| Scenario | Injected multiplier | Raw anchored ratio at week 80 | Table 9 |
|---|---:|---:|---:|
| Benefit | 0.8 | 0.824596370715 | 0.825 |
| Harm | 1.2 | 1.106983852125 | 1.107 |

For each dose and ISO week, sum deaths and people alive across rows. Compute the weekly hazard as `-log1p(-dead/alive)` and sum it from index 0 through the specified index, inclusive. Divide the dose-1 cumulative hazard by the dose-0 cumulative hazard at index 80, then divide by the same ratio at index 4. The generator injects the multiplier at indices 20 through 79. Index 80 includes one subsequent week. The [independent calculation](positive_check.py) reads the workbook and the two printed entries directly.

## Which execution produced what?

- **Raw workbook → plot and printed endpoints.** Running the shipped plotter on the supplied workbook selected its raw fallback and printed 1.1070 and 0.8246. That branch calls neither a frailty fit nor an inversion. The figure routine consumes the `weeks` and `kcor` arrays returned by the raw branch. The processed-output reader returns a different schema containing only `df` and `sheet_name`. See `test/positive_control/code/plot_positive_control.py`, lines 23–79, 111–151, 170–175 and 262–286.
- **Supplied workbook → production fitter.** The positive-control Makefile, lines 39–52, calls `code/KCOR.py` on the supplied workbook and passes its processed output to the plotter. Replaying the fitter command with temporary output paths exits 1 on the first sheet, `2021_24_benefit`. The parser expects two underscore-delimited fields and receives three. No processed workbook is created. All four editions failed before either scenario reached the fitter (m17 line 6666, m42 and m61 line 6734, m83 line 6740). The Makefile dependency prevents its figure recipe from running after this failure.
- **Historical table and figure.** The table contains literal entries in `documentation/preprint/paper.md`. All four editions contain the same two entries and byte-identical copies of the positive-control workbook, generator, plotting code, Makefile and PNG. The manuscript's PNG matches the positive-control analysis PNG. A raw replay recreated the results, but the locally rendered PNG differed in dimensions by one pixel. Neither historical fit logs nor a processed positive-control workbook accompany these artifacts. The numerical match and software failure establish a reproducibility gap. They do not establish how the authors originally obtained the table or figure.

## Review history and disposition

The original review questioned how well findings from a pulse separated from fitting windows would generalize. Series 2 explicitly observed the plotting fallback in an open baseline review and requested fitted provenance. Its follow-up reserved judgment on the historical branch. Series 3 independently linked the fallback numerically to both published entries, including on the previously evaluated edition. The issue predates iteration 2. [Original review, objection 4](../../../original-review/KCOR_Independent_Review_EN.md) · [Series 2 open review](../2/repeat-1/baseline.md) · [Series 2 follow-up](../2/repeat-2/B-follow.md) · [Series 3 previous edition](repeat-3/evaluated-first.md) · [Series 3 current edition](repeat-1/original-handoff.md)

**N6 is resolved as a description of the intervention.** SI §§S3.1 and S4.3 now correctly identify the treated cohort's own weekly hazard among survivors and the half-open injection window. **Fitted validation remains open.** Main §3.2 and SI §S3.1 still attribute the displayed result to the estimation pipeline. Label the reported values as raw comparators until a reproducible run supplies fitted endpoints, fitted variances and fit status for each cohort. [N6 decision](../../../2.6/README.md)

From this directory, run the following against any edition reconstructed by `freeze.py`. The arithmetic check uses Python and openpyxl. The optional replay uses the scientific edition's fitter and plotting dependencies, with ordinary default KCOR settings.

```powershell
py -3.11 -B positive_check.py "C:/Users/jasmi/AppData/Local/Temp/kcor-s3-8d29/m83" --replay
```
