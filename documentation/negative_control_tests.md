# Negative control tests

The idea is to show that we can compare cohorts of the same dose, but across different ages, and it should be close to a flat line. Since different ages have different mortality/frailty/comorbidities, this should be a reasonable sanity test.

The idea is to repurpose all the tooling we have in place and just generate a KCOR output that looks like the regular KCOR output so we can just "refresh" the database link and get the new data.

So the idea is to have a 

NEGATIVE_CONTROL_MODE =1

flag in the code which we can set to make this change.

If the flag is set the code will skip normal processing and just output data/negative_control_test_summary.xlsx file with 

Enrollment date     YoB1    Yob2      Dose    KCOR value at the normal reporting date for that cohort

Where the index values are the first 4 columns. so all the permutations where YoB1 runs through 1920 to 1980  and YoB2 is 10 years more than YoB1. And dose runs from 0 to 2 for the first two enrollments, and 0 to 3 for the other enrollments.

So the KCOR value is comparing the specific Dose on the two different years: “For each `EnrollmentDate` and `Dose`, compare YoB1 vs YoB2=YoB1+10 within the same sheet and dose.”


KCOR is the ratio of cumulative hazards (CHR) with baseline normalized at week 4, using the normal pipeline, and normal reporting date.

No need for slope normalization, Czech unvax MR correction, skip weeks, or anything else. 


Use these enrollments and doses: “2021_13, 2021_24: doses 0–2; 2022_06: 0–3; 2022_47: 0–4 

- Output spec:
  - File: `data/negative_control_test_summary.xlsx`
  - Sheet: one sheet with columns `EnrollmentDate, YoB1, YoB2, Dose, KCOR`, `CI_Lower, CI_Upper`.
  - Include units/format (e.g., YoB ints, KCOR to 4 decimals). 

