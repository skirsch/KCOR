KCOR time series

KCOR uses fixed cohorts defined on an enrollment date to analyze mortality in cohorts given dose 0, 1, 2, etc. of various ages.

I want to implement a complementary analysis that computes the mortality rate per week of cohorts defined relative to the time they got their shot.

So we'd read in the Czech data as we now do with KCOR_CMR, but let's create a new analysis called KCOR_ts for time series which reads in the same data, but processes it completely differently because we group the cohorts based on the time they got the shot instead of a fixed enrollment time.

CRITICAL: Each dose is processed independently as if it is the ONLY dose the person received. When processing dose N, ignore all other doses (both earlier and later). For example:
- Dose 1 cohort: Only consider Date_FirstDose, ignore Date_SecondDose, Date_ThirdDose, etc. Count weeks from Date_FirstDose.
- Dose 2 cohort: Only consider Date_SecondDose, ignore Date_FirstDose and all subsequent doses. Count weeks from Date_SecondDose.
- Dose 3 cohort: Only consider Date_ThirdDose, ignore all other doses. Count weeks from Date_ThirdDose.
- Similarly for doses 4 and 5.

So the output file would have columns for:

dose     Decade of Birth     week after dose      # alive        # dead    h(t)

Note that the first 3 columns are group-by index columns, the next two columns are the value columns (counts), and the last column is the continuous time hazard function. The weekly hazard h(t) is computed using -(ln(1 - dead/alive)) where dead/alive is the proportion of deaths among those alive at the start of that week.

For the index columns:
- Dose should vary from 1 to 5 (each dose processed independently)
- Decade of birth should vary from 1920 to 2000 (increment by 10)
- Week after dose should vary from 0 to 200

Processing notes:
- Process birth years from 1910 to 2005 inclusive (to properly compute decade groupings for output decades 1920-2000)
- Output format: Single Excel sheet
- No need for Sex or DCCI grouping dimensions at this time

Build:
- Run `make ts` from the code directory (or `make ts DATASET=Czech` from root)
- Output file: `data/Czech/KCOR_ts.xlsx` (or `data/$(DATASET)/KCOR_ts.xlsx` for other datasets)
- Uses the same input data structure as KCOR_CMR (records.csv)
- Follows the standard KCOR directory structure with DATASET=Czech as default
