# Sensitivity to quiet window tests


Perfect — that’s enough to do this cleanly off the Excel without touching `kcor.py`.

Yes: I’d make this a **separate analysis/validation script** (and optionally a notebook) that **reads `data/Czech/KCOR_CMR.xlsx`**, computes ((\hat k,\hat\theta)) under your same NLS objective, and writes a CSV + 1 figure for the SI. Keep `kcor.py` as the reference implementation.

Below is a **Cursor-ready punchlist** + a concrete implementation plan that matches your paper’s definitions.

Let's make the scripts and output in the test/quiet_window directory with similar file structure to the test/negative_control structure (code, data, analysis, out)

---

## What we will compute (per your paper)

Using the aggregated weekly data:

* Weekly risk ( \mathrm{MR}_t = d(t)/N(t) ) where:

  * (d(t) = \texttt{Dead})
  * (N(t) = \texttt{Alive}) (assumed “alive at start of week”)
* Discrete hazard: ( h(t) = -\log(1-\mathrm{MR}_t) )
* Cumulative hazard: ( H_{\mathrm{obs}}(t)=\sum_{s\le t} h(s) )
* Fit on bins whose **calendar ISOweek** lies in the quiet window:

  * Model in window: ( H^{model}(t)=\frac{1}{\theta}\log(1+\theta k t) )
  * NLS objective: minimize (\sum (H_{obs}(t)-H^{model}(t))^2)
  * Constraints: (k>0,\ \theta\ge 0)

Key detail: **(t) must be weeks since enrollment**, not “weeks since quiet window start.”
For `2021_24`, interpret enrollment ISOweek as **2021 week 24** (consistent with your naming).

---

## Quiet-window scan you proposed

* 12-month windows
* starts: April 2022 stepping monthly until April 2023
* For each window, fit theta for dose cohorts (at least dose 0, 1, 2 — whatever you want to report)
* Report stability especially for **Dose 0**.

This is absolutely persuasive if you also report **pass rate**.

---

## Cursor punchlist (no `kcor.py` changes)

### 1) Add a new script

Create:

* `scripts/quiet_window_scan_theta_czech_2021_24.py`

It should:

* load `data/Czech/KCOR_CMR.xlsx`, sheet `2021_24`
* filter strata: `YearOfBirth in {1930..1939, 1940..1949, 1950..1959}`
* group by: `YearOfBirth decade`, `Dose`, and `ISOweekDied`
* compute hazard + cumulative hazard per group
* compute `t = weeks_since_enrollment` using enrollment week = 2021-W24
* loop windows and fit ((k,\theta)) per group

### 2) Add a helper for ISOweek parsing

`ISOweekDied` might be formatted in one of several ways. Make the parser robust:

* `"YYYY_WW"` (e.g., `2022_14`)
* `"YYYY-WW"`
* `"YYYYWww"`
* `"YYYY-Www"`
* etc.

Convert to `(iso_year, iso_week)` → then to an absolute week index for subtraction.

### 3) Define the monthly window starts

Make windows based on calendar months, but convert to ISO weeks internally:

* Start dates: 2022-04-01, 2022-05-01, …, 2023-04-01
* For each start date, compute `start_iso_year, start_iso_week = date.isocalendar()`
* End date = start date + 12 months (use `dateutil.relativedelta(months=+12)` if you already have it; otherwise implement month add manually)
* Convert end date to ISO year/week too
* Quiet window in terms of absolute ISO-week index:

  * include weeks where `start_abs <= week_abs < end_abs`

### 4) Fit routine

Use a simple, stable constrained optimizer:

* `scipy.optimize.least_squares` with bounds:

  * `k in (1e-12, inf)`
  * `theta in [0, inf)` (use small positive lower bound like 0, and handle theta≈0 case carefully)

For theta near 0, use the limit:

* (H^{model}(t)\approx k t)

Compute:

* `rmse_H` over fit bins
* post-normalization linearity inside window:

  * compute (\tilde H_0(t) = (\exp(\theta H_{obs}(t)) - 1)/\theta) (limit (\tilde H_0=H_{obs}) for theta≈0)
  * regress (\tilde H_0(t)) vs (t) (within window) and record `R2` (or slope drift metric)
* `n_points` (# weeks used)

### 5) Diagnostics pass/fail (simple but aligned)

For this “quiet-window availability” demo, you don’t need every KCOR diagnostic — just a conservative subset:

Suggested pass criteria (tunable later):

* `n_points >= 30` (or whatever makes sense)
* `rmse_H <= rmse_thresh` (pick after looking at distribution; start with a percentile-based cutoff)
* `R2_postnorm >= 0.995` (or 0.99; again tune)
* `theta` not exploding (e.g., theta < 100)

Even better: report metrics continuously and show pass/fail based on thresholds. The figure becomes self-explanatory.

### 6) Output CSV

Write:

* `out/quiet_window_scan_theta_czech_2021_24_yob1930_40_50.csv`

Columns:

* `window_start_date`, `window_end_date`
* `window_start_iso`, `window_end_iso`
* `yob_decade` (1930/1940/1950)
* `dose`
* `k_hat`, `theta_hat`
* `rmse_H`, `r2_postnorm`, `n_points`
* `pass` (0/1)

Also write a one-line summary CSV:

* `out/quiet_window_scan_theta_summary.csv`
  with:
* pass rate per dose (and per decade)
* mean/SD/CV of theta among passing windows (esp dose 0)

### 7) Produce 1 SI figure (optional but high impact)

If you already have a plotting stack:

* x-axis: window midpoint date
* y-axis: `theta_hat`
* one panel per dose (or just dose 0 + overlay others)
* points colored by pass/fail
* separate lines for decades (or pooled)

Save:

* `figures/si/fig_quiet_window_theta_scan_czech_2021_24.png`

---

## Why this should be separate from `kcor.py`

* This is **validation tooling**, not core estimator code.
* You want to avoid any appearance that you “changed the method” to answer a reviewer.
* Your repo stays clean: `kcor.py` = method; `scripts/` = experiments/SI.

So: **yes** — separate script.

---

## One small question you don’t need to answer now (we can infer)

Is `Alive` in the spreadsheet the **risk set at the start of that ISO week**?
That’s the natural interpretation and matches your earlier definitions. If it’s instead “alive at end of week,” the hazard formula needs a minor tweak — but we can detect this quickly because hazards will misbehave (e.g., `Dead > Alive` or weird MR spikes).

---

If you want, I can now write the full script content (ready to paste into `scripts/quiet_window_scan_theta_czech_2021_24.py`) using pandas + scipy, with robust ISO-week parsing and clean outputs.
