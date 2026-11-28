# 5.12 - Round 5 finding

**Severity:** Moderate reproducibility

Quiet scan relies on changed default rather than explicit forced alpha; preprint Makefile references missing assets and publication target.

**Related issues:** 2.5, 4.7, documentation packaging

**Review evidence:** A B (build); A (scan) - [Series 5 convergence](../../iterative%20review/evaluations/5/convergence.md) ([A](../../iterative%20review/evaluations/5/blind-a.md), [B](../../iterative%20review/evaluations/5/blind-b.md), [R](../../iterative%20review/evaluations/5/resolution.md)).

**Status:** addressed in m117, preserving m116. The scan explicitly selects the marked wave (2021-41 through 2022-12) and alpha=1.18, checks the nonlinear fitter's reported law, and uses a separate `--mode off` output for the proportional fit. Both modes reproduce 78 attempted and 78 accepted fits. The forced-alpha fit reproduces all 78 saved frailty variances exactly and its plotted figure is byte-identical. [Numerical check](scan-verification.json), [forced results](windows.csv), [module-off results](windows-off.csv), [scan patch](scan.patch).

The local Makefile build now uses the bundled BibTeX bibliography and supplied [formatting assets](build-assets/). `make paper` and `make paper-pdf` compile from the included figures. The empirical intuition figure has an explicit regeneration target, and publication has the separate `make paper-publish` target. Updated documentation maps Table 8, Table 9 and the scan to their stored results. [Makefile patch](Makefile.patch), [documentation patch](analysis-readme.patch). Scientific manuscript wording and the fitted method are unchanged by this revision.

Verification: forced local `make -B paper` completed successfully on Windows using GNU Make, Git sh, Pandoc 3.12.1, pandoc-crossref 0.3.26, XeLaTeX, Times New Roman and Cambria Math. Dry runs confirmed that local compilation invokes neither Python figure generation nor publication; the separate publication target retains the original command. [Build log](build-verification.txt), [Makefile-generated manuscript](make-paper.pdf). The normal Makefile table page-break filter yields 75 pages. The 69-page [review manuscript](KCOR-clean.pdf) and [round-5 redline](KCOR-redline.pdf) are unchanged from m116.
