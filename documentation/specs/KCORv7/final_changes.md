## final changes

Here they are:

---

**Fix 1: Lower relRMSE bad-fit threshold from 1e6 to 1e5**

In `code/KCOR.py`, change the bad-fit relRMSE threshold constant from `1e6` to `1e5`. If it's in a config file (`Czech.yaml` or equivalent), update the default there instead. No other changes — one number, one line.

Validation: confirm `2022_06 YOB=1990 dose=2` now shows `theta_applied=0` and `theta_fit_status='insufficient_signal'`, and that no new `ok` cases have `relRMSE > 1e5`.

---

**Fix 2: Make `any_non_ok_status` meaningful by scoping it to well-identified cohorts**

Currently `any_non_ok_status=True` on every enrollment date because young/low-mortality cohorts always produce non-OK statuses. This makes the flag useless as a quality signal.

In `create_summary_file` in `code/KCOR.py`, replace the current `any_non_ok_status` logic with two separate columns:

`any_non_ok_status` — keep as-is for full transparency, always reflects whether any cohort in the enrollment date has a non-OK status.

`quality_flag` — new column, `True` only when a **well-identified cohort** has a non-OK status. Define well-identified as: `year_of_birth <= 1960` AND `total_quiet_deaths >= min_quiet_deaths` AND `dose == 0`. These are the older high-mortality unvaccinated cohorts where a non-OK status is unexpected and meaningful. If all non-OK statuses come from young cohorts or vaccinated cohorts with low deaths, `quality_flag` stays `False`.

Also add a companion column `quality_flag_detail` — a comma-separated string listing the specific `(year_of_birth, dose, status)` tuples that triggered `quality_flag=True`, so the cause is immediately visible without opening the full diagnostics sheet. Leave it empty string when `quality_flag=False`.

Add both columns to the `theta0_status_summary` sheet in `KCOR_summary.xlsx` and to the `dual_print` log summary so they appear in the run log.

Validation:
- Confirm `quality_flag=False` for all enrollment dates on the Czech data — none of the non-OK cases should be well-identified cohorts
- Confirm `quality_flag_detail` is empty string for all rows
- Confirm `any_non_ok_status` remains `True` as before for all rows
- Run lints on `code/KCOR.py`
