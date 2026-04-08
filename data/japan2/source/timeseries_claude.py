import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Pre-specify ALL analytical choices before running
DATA_FILE    = 'records.csv'
BIRTH_YEAR_MIN = 1930  # cohort definition
BIRTH_YEAR_MAX = 1960
MAX_WEEKS    = 104     # 2 years observation window
HVE_WEEKS    = 2       # HVE window per empirical evidence
MIN_DOSE     = 2       # minimum dose number to include
QUIET_PERIODS = [
    ('2021-06-14', '2021-10-01'),  # 2021 quiet period
    ('2023-05-08', '2023-10-01'),  # 2023 quiet period
]

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_FILE, dtype=str, na_values=['', 'nan', '-1'])
print(f"Loaded {len(df):,} records")

# ── 2. PARSE YEAR-WEEK DATES ──────────────────────────────────────────────────
def parse_yw(s):
    try:
        year, week = str(s).strip().split('-')
        return pd.Timestamp.fromisocalendar(int(year), int(week), 1)
    except:
        return pd.NaT

date_cols = ['Date_FirstDose','Date_SecondDose','Date_ThirdDose',
             'Date_FourthDose','Date_FifthDose','Date_SixthDose',
             'DateOfDeath']

print("Parsing dates...")
for col in date_cols:
    df[col] = df[col].apply(parse_yw)

# ── 3. PARSE BIRTH YEAR ───────────────────────────────────────────────────────
def parse_birth_year(s):
    # Format is "1940-1944" or "1945-1949" etc
    try:
        return int(str(s).split('-')[0])
    except:
        return None

df['BirthYearMin'] = df['YearOfBirth'].apply(parse_birth_year)

# ── 4. FILTER COHORT ──────────────────────────────────────────────────────────
cohort = df[
    df['BirthYearMin'].notna() &
    (df['BirthYearMin'] >= BIRTH_YEAR_MIN) &
    (df['BirthYearMin'] <= BIRTH_YEAR_MAX) &
    df['Date_FirstDose'].notna()
].copy()

print(f"Cohort (born {BIRTH_YEAR_MIN}-{BIRTH_YEAR_MAX}, vaccinated): {len(cohort):,}")
print(f"Deceased in cohort: {cohort['DateOfDeath'].notna().sum():,}")

# ── 5. LAST DOSE DATE AND NUMBER ─────────────────────────────────────────────
dose_cols = ['Date_FirstDose','Date_SecondDose','Date_ThirdDose',
             'Date_FourthDose','Date_FifthDose','Date_SixthDose']

def last_dose_info(row):
    last_date = pd.NaT
    last_num  = 0
    death_date = row['DateOfDeath']
    for i, col in enumerate(dose_cols, 1):
        if pd.notna(row[col]):
            # If deceased, only consider doses that occurred before death
            if pd.notna(death_date) and row[col] >= death_date:
                continue
            last_date = row[col]
            last_num  = i
    return last_date, last_num

print("Computing last dose info...")
results = cohort.apply(last_dose_info, axis=1)
cohort['LastDoseDate'] = [r[0] for r in results]
cohort['LastDoseNum']  = [r[1] for r in results]

# Apply MIN_DOSE filter
cohort = cohort[cohort['LastDoseNum'] >= MIN_DOSE].copy()
print(f"Cohort after MIN_DOSE>={MIN_DOSE} filter: {len(cohort):,}")

# ── 6. PERSON-WEEKS DENOMINATOR ───────────────────────────────────────────────
# For each person, they contribute 1 person-week to every week
# from 1 to min(weeks_until_death_or_censoring, MAX_WEEKS)
# Censoring date = end of observation period (use max death date in dataset)
CENSOR_DATE = df['DateOfDeath'].max()
print(f"Censoring date: {CENSOR_DATE}")

def weeks_at_risk(row):
    start = row['LastDoseDate']
    if pd.isna(start):
        return 0
    if pd.notna(row['DateOfDeath']):
        end = row['DateOfDeath']
    else:
        end = CENSOR_DATE
    weeks = max(0, (end - start).days // 7)
    return min(weeks, MAX_WEEKS)

print("Computing person-weeks at risk...")
cohort['WeeksAtRisk'] = cohort.apply(weeks_at_risk, axis=1)

# Build denominator array
person_weeks = np.zeros(MAX_WEEKS + 1)
for w in cohort['WeeksAtRisk']:
    if w > 0:
        person_weeks[1:int(w)+1] += 1

print(f"Total person-weeks at risk: {person_weeks.sum():,.0f}")

# ── 7. DECEASED ONLY: WEEKS SINCE LAST DOSE ───────────────────────────────────
deceased = cohort[cohort['DateOfDeath'].notna()].copy()
deceased['DaysSinceLastDose'] = (
    deceased['DateOfDeath'] - deceased['LastDoseDate']
).dt.days
deceased['WeeksSinceLastDose'] = deceased['DaysSinceLastDose'] // 7

# Keep positive, within window
deceased = deceased[
    (deceased['DaysSinceLastDose'] > 0) &
    (deceased['WeeksSinceLastDose'] <= MAX_WEEKS)
]
print(f"Deceased with positive days since last dose: {len(deceased):,}")

# ── 8. MORTALITY RATE BY WEEK SINCE LAST DOSE ─────────────────────────────────
weekly_deaths = deceased.groupby('WeeksSinceLastDose').size()
weekly_deaths = weekly_deaths.reindex(range(1, MAX_WEEKS+1), fill_value=0)

# Rate = deaths / person-weeks at risk
weekly_rate = pd.Series(index=range(1, MAX_WEEKS+1), dtype=float)
for w in range(1, MAX_WEEKS+1):
    if person_weeks[w] > 0:
        weekly_rate[w] = weekly_deaths[w] / person_weeks[w]
    else:
        weekly_rate[w] = np.nan

# ── 9. BY DOSE NUMBER ─────────────────────────────────────────────────────────
weekly_by_dose = deceased.groupby(
    ['WeeksSinceLastDose','LastDoseNum']
).size().unstack(fill_value=0)

# ── 10. ALL-DOSES ANALYSIS ────────────────────────────────────────────────────
# Each person counted once per dose received before death
print("\nComputing all-doses analysis...")
all_deaths_arr  = np.zeros(MAX_WEEKS + 1)
all_pw_arr      = np.zeros(MAX_WEEKS + 1)
all_by_dosenum  = {}

for i, col in enumerate(dose_cols, 1):
    sub = cohort[cohort[col].notna()].copy()
    # Only doses before death (or person is alive)
    sub = sub[sub['DateOfDeath'].isna() | (sub[col] < sub['DateOfDeath'])].copy()

    # Denominator: person-weeks from this dose date to death/censor
    def _war(row, c=col):
        start = row[c]
        end = row['DateOfDeath'] if pd.notna(row['DateOfDeath']) else CENSOR_DATE
        return min(max(0, (end - start).days // 7), MAX_WEEKS)
    sub['_war'] = sub.apply(_war, axis=1)
    for w in sub['_war']:
        if w > 0:
            all_pw_arr[1:int(w)+1] += 1

    # Numerator: deceased — weeks from this dose to death
    dec = sub[sub['DateOfDeath'].notna()].copy()
    dec['_days'] = (dec['DateOfDeath'] - dec[col]).dt.days
    dec['_wk']   = dec['_days'] // 7
    dec = dec[(dec['_days'] > 0) & (dec['_wk'] >= 1) & (dec['_wk'] <= MAX_WEEKS)]
    for w in dec['_wk']:
        all_deaths_arr[int(w)] += 1
    all_by_dosenum[i] = dec.groupby('_wk').size().reindex(
        range(1, MAX_WEEKS+1), fill_value=0)

all_deaths_series = pd.Series(all_deaths_arr[1:MAX_WEEKS+1], index=range(1, MAX_WEEKS+1))
all_rate = pd.Series(index=range(1, MAX_WEEKS+1), dtype=float)
for w in range(1, MAX_WEEKS+1):
    all_rate[w] = all_deaths_arr[w] / all_pw_arr[w] if all_pw_arr[w] > 0 else np.nan

print(f"All-doses: {int(all_deaths_arr.sum()):,} death-dose observations")

# ── 11. PLOTS — LAST DOSE ─────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(16, 16))

# Panel 1: Raw death counts
axes[0].bar(weekly_deaths.index, weekly_deaths.values,
            width=0.8, color='steelblue', alpha=0.7)
axes[0].axvline(x=HVE_WEEKS, color='red', linestyle='--',
                linewidth=2, label=f'{HVE_WEEKS}-week HVE window')
axes[0].set_title(
    f'Deaths by weeks since last dose — cohort born {BIRTH_YEAR_MIN}-{BIRTH_YEAR_MAX}',
    fontsize=12)
axes[0].set_xlabel('Weeks since last dose')
axes[0].set_ylabel('Death count')
axes[0].legend()

# Panel 2: Normalized mortality rate
axes[1].plot(weekly_rate.index, weekly_rate.values,
             color='darkorange', linewidth=1.5)
axes[1].axvline(x=HVE_WEEKS, color='red', linestyle='--',
                linewidth=2, label=f'{HVE_WEEKS}-week HVE window')
axes[1].set_title('Mortality rate by weeks since last dose (person-week normalized)',
                  fontsize=12)
axes[1].set_xlabel('Weeks since last dose')
axes[1].set_ylabel('Deaths per person-week')
axes[1].legend()

# Panel 3: By dose number
for dose_num in sorted(weekly_by_dose.columns):
    axes[2].plot(weekly_by_dose.index, weekly_by_dose[dose_num],
                label=f'Last dose: {int(dose_num)}', linewidth=1.5)
axes[2].axvline(x=HVE_WEEKS, color='red', linestyle='--',
                linewidth=2, label=f'{HVE_WEEKS}-week HVE window')
axes[2].set_title('Deaths by weeks since last dose, by last dose number',
                  fontsize=12)
axes[2].set_xlabel('Weeks since last dose')
axes[2].set_ylabel('Deaths')
axes[2].legend()

plt.tight_layout()
plt.savefig('timeseries_last_dose.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart saved: timeseries_last_dose.png")

# ── 12. PLOTS — ALL DOSES (each person counted once per dose) ─────────────────
fig, axes = plt.subplots(3, 1, figsize=(16, 16))

axes[0].bar(all_deaths_series.index, all_deaths_series.values,
            width=0.8, color='steelblue', alpha=0.7)
axes[0].axvline(x=HVE_WEEKS, color='red', linestyle='--',
                linewidth=2, label=f'{HVE_WEEKS}-week HVE window')
axes[0].set_title(
    f'Deaths by weeks since each dose — cohort born {BIRTH_YEAR_MIN}-{BIRTH_YEAR_MAX} (all doses)',
    fontsize=12)
axes[0].set_xlabel('Weeks since dose')
axes[0].set_ylabel('Death count')
axes[0].legend()

axes[1].plot(all_rate.index, all_rate.values, color='darkorange', linewidth=1.5)
axes[1].axvline(x=HVE_WEEKS, color='red', linestyle='--',
                linewidth=2, label=f'{HVE_WEEKS}-week HVE window')
axes[1].set_title('Mortality rate by weeks since each dose (person-week normalized, all doses)',
                  fontsize=12)
axes[1].set_xlabel('Weeks since dose')
axes[1].set_ylabel('Deaths per person-week')
axes[1].legend()

for dose_num in sorted(all_by_dosenum.keys()):
    axes[2].plot(all_by_dosenum[dose_num].index, all_by_dosenum[dose_num].values,
                label=f'Dose: {dose_num}', linewidth=1.5)
axes[2].axvline(x=HVE_WEEKS, color='red', linestyle='--',
                linewidth=2, label=f'{HVE_WEEKS}-week HVE window')
axes[2].set_title('Deaths by weeks since each dose, by dose number',
                  fontsize=12)
axes[2].set_xlabel('Weeks since dose')
axes[2].set_ylabel('Deaths')
axes[2].legend()

plt.tight_layout()
plt.savefig('timeseries_all_doses.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart saved: timeseries_all_doses.png")

# ── 13. ROLLOUT TIMELINE ──────────────────────────────────────────────────────
# Cumulative people vaccinated per dose number over calendar time (full dataset)
print("\nComputing rollout timeline...")
rollout_start = pd.Timestamp('2021-01-01')
rollout_end   = CENSOR_DATE
date_range    = pd.date_range(start=rollout_start, end=rollout_end, freq='7D')

rollout = {}
for i, col in enumerate(dose_cols, 1):
    dose_dates = df[col].dropna().sort_values()
    if len(dose_dates) == 0:
        rollout[i] = pd.Series(0, index=date_range)
        continue
    weekly_counts = dose_dates.value_counts().sort_index().cumsum()
    rollout[i] = weekly_counts.reindex(date_range, method='ffill').fillna(0)

fig, ax = plt.subplots(figsize=(16, 6))
colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown']
for i in range(1, len(dose_cols)+1):
    ax.plot(rollout[i].index, rollout[i].values,
            label=f'Dose {i}', linewidth=1.5, color=colors[i-1])
ax.set_title('Vaccination rollout — cumulative recipients by dose number (full dataset)',
             fontsize=12)
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative people vaccinated')
ax.legend()
ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('timeseries_rollout.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart saved: timeseries_rollout.png")

# ── 14. UNVACCINATED DEATHS BY REFERENCE DATE ─────────────────────────────────
# For each monthly reference date Mar-Dec 2021, take people unvaccinated as of
# that date (within the cohort), and plot deaths by week since that date.
print("\nComputing unvaccinated deaths by reference date...")

ref_dates = pd.date_range(start='2021-03-01', end='2021-12-01', freq='MS')

fig, ax = plt.subplots(figsize=(16, 8))
colors14 = plt.cm.tab10(np.linspace(0, 1, len(ref_dates)))

for ref_date, color in zip(ref_dates, colors14):
    # Cohort members unvaccinated as of ref_date
    unvacc = df[
        df['BirthYearMin'].notna() &
        (df['BirthYearMin'] >= BIRTH_YEAR_MIN) &
        (df['BirthYearMin'] <= BIRTH_YEAR_MAX) &
        (df['Date_FirstDose'].isna() | (df['Date_FirstDose'] > ref_date))
    ].copy()

    # Deceased after ref_date
    died = unvacc[
        unvacc['DateOfDeath'].notna() &
        (unvacc['DateOfDeath'] > ref_date)
    ].copy()
    died['WeeksSinceRef'] = (died['DateOfDeath'] - ref_date).dt.days // 7
    died = died[(died['WeeksSinceRef'] >= 1) & (died['WeeksSinceRef'] <= MAX_WEEKS)]

    weekly = died.groupby('WeeksSinceRef').size().reindex(
        range(1, MAX_WEEKS+1), fill_value=0)
    ax.plot(weekly.index, weekly.values,
            label=ref_date.strftime('%b %Y'), linewidth=1.5, color=color)

ax.set_title(
    f'Deaths among unvaccinated — cohort born {BIRTH_YEAR_MIN}-{BIRTH_YEAR_MAX}, by reference date',
    fontsize=12)
ax.set_xlabel('Weeks since reference date')
ax.set_ylabel('Deaths')
ax.legend(title='Unvacc. as of:')
plt.tight_layout()
plt.savefig('timeseries_unvaccinated.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart saved: timeseries_unvaccinated.png")

# ── 15. CALENDAR DEATHS + MORTALITY RATE FOR ENTIRE COHORT ───────────────────
# Uses everyone born BIRTH_YEAR_MIN-BIRTH_YEAR_MAX (vaccinated and unvaccinated)
print("\nComputing calendar-time mortality for entire cohort...")

full_cohort = df[
    df['BirthYearMin'].notna() &
    (df['BirthYearMin'] >= BIRTH_YEAR_MIN) &
    (df['BirthYearMin'] <= BIRTH_YEAR_MAX)
].copy()

total_cohort_size = len(full_cohort)
print(f"Full cohort size (born {BIRTH_YEAR_MIN}-{BIRTH_YEAR_MAX}): {total_cohort_size:,}")

# Weekly calendar grid covering full observation period
cal_start = pd.Timestamp('2021-01-04')   # first Monday of 2021
cal_weeks = pd.date_range(start=cal_start, end=CENSOR_DATE, freq='7D')

# Deaths per week (dates are already week-precision)
death_dates = full_cohort['DateOfDeath'].dropna()
weekly_deaths_cal = death_dates.groupby(death_dates).count()
weekly_deaths_cal = weekly_deaths_cal.reindex(cal_weeks, fill_value=0)

# People alive at start of each week = total - cumulative deaths before that week
cum_deaths_before = weekly_deaths_cal.cumsum().shift(1).fillna(0)
alive_per_week = total_cohort_size - cum_deaths_before

# Mortality rate = deaths this week / alive at start of week
mortality_rate_cal = (weekly_deaths_cal / alive_per_week).replace([np.inf, np.nan], np.nan)

fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

axes[0].bar(weekly_deaths_cal.index, weekly_deaths_cal.values,
            width=5, color='steelblue', alpha=0.7)
axes[0].set_title(
    f'Weekly deaths — entire cohort born {BIRTH_YEAR_MIN}-{BIRTH_YEAR_MAX} (n={total_cohort_size:,})',
    fontsize=12)
axes[0].set_ylabel('Deaths per week')

axes[1].plot(mortality_rate_cal.index, mortality_rate_cal.values,
             color='darkorange', linewidth=1.5)
axes[1].set_title('Weekly mortality rate — deaths / people alive that week', fontsize=12)
axes[1].set_xlabel('Calendar date')
axes[1].set_ylabel('Deaths per person per week')

for ax in axes:
    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig('timeseries_calendar.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart saved: timeseries_calendar.png")

# ── 16. CUMULATIVE HAZARD — UNVACCINATED, EVERY OTHER MONTH ──────────────────
# H(t) = -log(S(t)), S(t) = KM product Π(1 - d_w/n_w)
# 6 reference dates: Jan, Mar, May, Jul, Sep, Nov 2021
print("\nComputing cumulative hazard for unvaccinated cohort...")

ref_dates_hz = pd.date_range(start='2021-01-01', end='2021-11-01', freq='2MS')

fig, ax = plt.subplots(figsize=(16, 8))
colors_hz = plt.cm.tab10(np.linspace(0, 1, len(ref_dates_hz)))

for ref_date, color in zip(ref_dates_hz, colors_hz):
    # Unvaccinated AND alive as of ref_date
    unvacc_hz = df[
        df['BirthYearMin'].notna() &
        (df['BirthYearMin'] >= BIRTH_YEAR_MIN) &
        (df['BirthYearMin'] <= BIRTH_YEAR_MAX) &
        (df['Date_FirstDose'].isna() | (df['Date_FirstDose'] > ref_date)) &
        (df['DateOfDeath'].isna() | (df['DateOfDeath'] > ref_date))
    ].copy()

    n_start = len(unvacc_hz)
    if n_start == 0:
        continue

    # Weeks from ref_date to death (NaN = censored/alive)
    died_after = unvacc_hz['DateOfDeath'].notna()
    wk_to_death = ((unvacc_hz.loc[died_after, 'DateOfDeath'] - ref_date)
                   .dt.days // 7)
    wk_to_death = wk_to_death[(wk_to_death >= 1) & (wk_to_death <= MAX_WEEKS)]
    death_counts = np.bincount(wk_to_death.astype(int), minlength=MAX_WEEKS + 1)

    # H(t) = cumsum(-log(1 - MR))  per KCOR.py convention
    n_at_risk = n_start
    cum_H = 0.0
    H_vals = []
    for w in range(1, MAX_WEEKS + 1):
        d_w = death_counts[w]
        if n_at_risk > 0:
            mr = min(d_w / n_at_risk, 1.0 - 1e-10)
            cum_H += -np.log1p(-mr)
        H_vals.append(cum_H)
        n_at_risk -= d_w

    H_series = pd.Series(H_vals, index=range(1, MAX_WEEKS + 1))
    ax.plot(H_series.index, H_series.values,
            label=f'{ref_date.strftime("%b %Y")} (n={n_start:,})',
            linewidth=1.5, color=color)

ax.set_title(
    f'Cumulative hazard H(t) — unvaccinated cohort born {BIRTH_YEAR_MIN}-{BIRTH_YEAR_MAX}, by reference date',
    fontsize=12)
ax.set_xlabel('Weeks since reference date')
ax.set_ylabel('H(t) = −log(S(t))')
ax.legend(title='Unvacc. as of:')
plt.tight_layout()
plt.savefig('timeseries_hazard_unvacc.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart saved: timeseries_hazard_unvacc.png")

# ── HELPERS FOR DOSE-STRATIFIED HAZARD ───────────────────────────────────────
def get_dose_group(dataframe, enroll_date, dose_num):
    """People alive on enroll_date with exactly dose_num doses received by then."""
    base = dataframe[
        dataframe['BirthYearMin'].notna() &
        (dataframe['BirthYearMin'] >= BIRTH_YEAR_MIN) &
        (dataframe['BirthYearMin'] <= BIRTH_YEAR_MAX) &
        (dataframe['DateOfDeath'].isna() | (dataframe['DateOfDeath'] > enroll_date))
    ]
    if dose_num == 0:
        return base[base['Date_FirstDose'].isna() | (base['Date_FirstDose'] > enroll_date)].copy()
    this_col = dose_cols[dose_num - 1]
    mask = base[this_col].notna() & (base[this_col] <= enroll_date)
    if dose_num < len(dose_cols):
        next_col = dose_cols[dose_num]
        mask = mask & (base[next_col].isna() | (base[next_col] > enroll_date))
    return base[mask].copy()

def plot_hazard_by_dose(dataframe, enroll_date, dose_nums, filename):
    fig, ax = plt.subplots(figsize=(16, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(dose_nums)))
    for dose_num, color in zip(dose_nums, colors):
        group = get_dose_group(dataframe, enroll_date, dose_num)
        if len(group) == 0:
            print(f"  Dose {dose_num}: no members, skipping")
            continue
        H = km_cumulative_hazard(group, enroll_date, MAX_WEEKS)
        lbl = f'Dose {dose_num} (n={len(group):,})' if dose_num > 0 else f'Unvaccinated / Dose 0 (n={len(group):,})'
        ax.plot(H.index, H.values, label=lbl, linewidth=2, color=color)
    ax.set_title(
        f'Cumulative hazard H(t) by dose at enrollment — cohort born {BIRTH_YEAR_MIN}-{BIRTH_YEAR_MAX}, '
        f'enrolled {enroll_date.strftime("%b %d %Y")}',
        fontsize=12)
    ax.set_xlabel(f'Weeks since {enroll_date.strftime("%b %d %Y")}')
    ax.set_ylabel('H(t) = −log(S(t))')
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Chart saved: {filename}")

# ── 17. H(t) VACCINATED vs UNVACCINATED — ENROLLED NOV 1 2021 ────────────────
# Everyone alive on Nov 1 2021 enrolled; split by vax status on that date.
# Both groups followed forward from the same calendar anchor — no time confounding.
print("\nComputing H(t) vaccinated vs unvaccinated from Nov 1 2021...")

ENROLL_DATE = pd.Timestamp('2021-11-01')

base = df[
    df['BirthYearMin'].notna() &
    (df['BirthYearMin'] >= BIRTH_YEAR_MIN) &
    (df['BirthYearMin'] <= BIRTH_YEAR_MAX) &
    (df['DateOfDeath'].isna() | (df['DateOfDeath'] > ENROLL_DATE))
].copy()

vacc_enrolled   = base[base['Date_FirstDose'].notna() & (base['Date_FirstDose'] <= ENROLL_DATE)]
unvacc_enrolled = base[base['Date_FirstDose'].isna() | (base['Date_FirstDose'] > ENROLL_DATE)]

def km_cumulative_hazard(group, ref_date, max_weeks):
    """Return H(t) = cumsum(-log(1 - MR)) per KCOR.py hazard_from_mr convention.
    MR_w = d_w / n_w (deaths / at-risk at start of week w).
    H(t) = Σ_{w=1}^{t} -log(1 - MR_w)
    """
    n_start = len(group)
    if n_start == 0:
        return pd.Series(dtype=float)
    died_after = group['DateOfDeath'].notna()
    wk = ((group.loc[died_after, 'DateOfDeath'] - ref_date).dt.days // 7)
    wk = wk[(wk >= 1) & (wk <= max_weeks)].astype(int)
    death_counts = np.bincount(wk, minlength=max_weeks + 1)
    n_at_risk = n_start
    cum_H = 0.0
    H_vals = []
    for w in range(1, max_weeks + 1):
        d_w = death_counts[w]
        if n_at_risk > 0:
            mr = d_w / n_at_risk
            mr = min(mr, 1.0 - 1e-10)   # numerical guard
            cum_H += -np.log1p(-mr)      # -log(1 - MR)
        H_vals.append(cum_H)
        n_at_risk -= d_w
    return pd.Series(H_vals, index=range(1, max_weeks + 1))

H_vacc   = km_cumulative_hazard(vacc_enrolled,   ENROLL_DATE, MAX_WEEKS)
H_unvacc = km_cumulative_hazard(unvacc_enrolled, ENROLL_DATE, MAX_WEEKS)

fig, ax = plt.subplots(figsize=(16, 7))
ax.plot(H_vacc.index,   H_vacc.values,   color='tab:blue',
        linewidth=2, label=f'Vaccinated (n={len(vacc_enrolled):,})')
ax.plot(H_unvacc.index, H_unvacc.values, color='tab:orange',
        linewidth=2, label=f'Unvaccinated (n={len(unvacc_enrolled):,})')
ax.set_title(
    f'Cumulative hazard H(t) — cohort born {BIRTH_YEAR_MIN}-{BIRTH_YEAR_MAX}, '
    f'enrolled {ENROLL_DATE.strftime("%b %d %Y")} by vaccination status',
    fontsize=12)
ax.set_xlabel('Weeks since Nov 1 2021')
ax.set_ylabel('H(t) = −log(S(t))')
ax.legend()
plt.tight_layout()
plt.savefig('timeseries_hazard_vacc_vs_unvacc.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart saved: timeseries_hazard_vacc_vs_unvacc.png")

# ── 18. H(t) BY DOSE — ENROLLED JUN 1 2022 ───────────────────────────────────
print("\nComputing H(t) by dose, enrolled Jun 1 2022...")
plot_hazard_by_dose(df, pd.Timestamp('2022-06-01'), [0, 2, 3],
                    'timeseries_hazard_jun2022.png')

# ── 19. H(t) BY DOSE — ENROLLED MAR 1 2023 ───────────────────────────────────
print("Computing H(t) by dose, enrolled Mar 1 2023...")
plot_hazard_by_dose(df, pd.Timestamp('2023-03-01'), [0, 2, 3, 4, 5],
                    'timeseries_hazard_mar2023.png')

# ── 20. STATISTICAL TESTS ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("STATISTICAL ANALYSIS")
print("="*60)

# Test 1: HVE — is rate lower in first 2 weeks?
early_rate = weekly_rate[1:HVE_WEEKS+1].mean()
post_rate  = weekly_rate[HVE_WEEKS+1:13].mean()
print(f"\nHVE Test:")
print(f"Mean rate weeks 1-{HVE_WEEKS}: {early_rate:.8f}")
print(f"Mean rate weeks {HVE_WEEKS+1}-12: {post_rate:.8f}")
print(f"Ratio post/early: {post_rate/early_rate:.3f}")
print(f"HVE predicts ratio > 1 (early suppressed below post)")

# Test 2: Gompertz — after HVE window, is rate rising or falling?
post_hve_rate = weekly_rate[HVE_WEEKS+1:53].dropna()
log_rate = np.log(post_hve_rate)
t = np.arange(len(log_rate))

slope, intercept, r, p, se = stats.linregress(t, log_rate)
gamma_annual = slope * 52

print(f"\nGompertz Test (weeks {HVE_WEEKS+1}-52 post last dose):")
print(f"Slope gamma: {slope:.6f}/week = {gamma_annual:.1f}%/year")
print(f"SE: {se:.6f}")
print(f"t-stat: {slope/se:.3f}")
print(f"p-value: {p:.6f}")
print(f"R²: {r**2:.4f}")
print(f"Biological Gompertz gamma: ~8.5%/year")
print(f"Ratio observed/biological: {gamma_annual/8.5:.2f}x")
print()
if slope > 0 and p < 0.05:
    print("RESULT: Rate INCREASING after HVE window — p<0.05")
    print("Consistent with ongoing vaccine harm")
    print("Inconsistent with dynamic HVE (which predicts declining rate)")
elif slope < 0 and p < 0.05:
    print("RESULT: Rate DECREASING after HVE window — p<0.05")
    print("Consistent with dynamic HVE or survivor selection")
    print("NOT consistent with ongoing vaccine harm")
else:
    print("RESULT: No significant trend after HVE window — p={p:.3f}")
    print("Rate is essentially flat post-HVE")

# Test 3: Dose-response
print(f"\nDose-Response Test:")
print(f"Last dose number distribution among deceased:")
print(deceased['LastDoseNum'].value_counts().sort_index())

# ── 21. SAVE RESULTS ──────────────────────────────────────────────────────────
weekly_rate.to_csv('weekly_mortality_rate.csv', header=['rate'])
weekly_by_dose.to_csv('weekly_by_dose.csv')
print("\nResults saved to CSV files")
print("\nDone.")
