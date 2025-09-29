All slopes are per week; 0.002/week ≈ 10%/yr.


The Czech Republic didn't do a perfect job tracking the unvaccinated (since they never came in to decline vaccination). 1.6M of these unvaccinated didn't have a date of birth.

This caused the mortality rate of the unvaccinated with known dates of birth to be distorted, sometimes causing a negative slope over time.

To adjust for this, we ensure that the unvaccinated WEEKLY MORTALITY RATES are slope normalized using an adjustment slope computed from the slope difference of the dose 1 cohort and the dose 0 cohort computed using the standard 2 anchor slope method.

This adjustment is ONLY enabled if CZECH_UNVACCINATED_MR_ADJUSTMENT=1 is set (the default).

The slope normalization, which is ONLY done for the unvaccinated, is done at the MR level, creating an MR_adjusted just as we have now. It is done for all values after t_e (enrollment time), just as slope adjustment is done now.

```
If CZECH_UNVACCINATED_MR_ADJUSTMENT==1:
    adj = min(0.002, slope_d1) − slope_d0
    If adj > 0: for Dose=0, set MR_adj = MR * exp(+adj * (t − t_e))
```


The min keeps the mortality rate within "sane" limits after adjustment (i.e., the final slope will be less than 10%/yr).

The slope adjustment is performed, regardless of the setting of SLOPE_NORMALIZE_YOB_LE value, only on the Dose 0 cohort when slope_adjust_value is positive. 

If normal slope adjustment applies, that will entirely supersede the CZECH setting. 

When the CZECH adjustment is enabled, will always INCREASE the slope of the MR(t) over time only for the dose 0 group. The log notes the value on the line for that YoB:

Log Before (for Dose 0)
 YoB 1920, Dose 0: slope = 0.003529

Log After (only if there is a change):
 YoB 1920, Dose 0: slope = 0.003529, adjusting by +.0002344 per week (Czech data issue)
