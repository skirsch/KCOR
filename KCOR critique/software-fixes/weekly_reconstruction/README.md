# Exact weekly reconstruction in the KCOR fitter

The fitter uses a first-order approximation when reconstructing cumulative hazard between quiet windows. A small downward error can make the reconstructed excess at a gap nonpositive. The first-pass rule then sets frailty to zero and flags insufficient signal.

[The software patch](exact_weekly_update.patch) replaces the two internal updates with the exact formula obtained by subtracting the gamma-frailty cumulative inversion at successive weeks. It uses `expm1` for the exponential difference and the observed weekly increment at zero frailty. It retains the zero-offset rule, the baseline fit, and the final cumulative inversion.

The same two substitutions were tested in memory on 720 paired simulated datasets and 108 saved Czech cohort fits. They changed decisions in 28 simulated comparisons and zero replayed cohort fits. Among the 33 Czech comparisons that passed both ways, the maximum absolute KCOR change was 0.000562. [Design and results](<../../objections/1.6/objection6_weekly_inversion_multiwindow/README.md>)

The patch passed `git apply --check` in the KCOR repository. The companion manuscript patch describes the exact reconstruction and belongs with its adoption. [Objection 6](<../../objections/1.6/README.md>)
