# Slope Normalization Method #2 (aka slope2)

The slope2 method is very simple.

You have two windows: W1 and W2. These are ideally chosen in quiet periods a year apart from each other.

You take the mean of the hazards in each window: Wm1 and Wm2. Do NOT take the geometric mean since low values can distort it.

To compute the slope of these values:

beta= (ln(Wm2)-ln(Wm1))/(center date W2  - Center date W1).

 That gives you β. Center date can be computed as a regular date because I adjusted all the dates for even number of days. If not, give an error and stop.

Coverage gaps: Treat all missing hazard values as hazard=0 for that date.

You then apply the opposite of that slope as a exponential multiplier to the cohort starting from enrollment (so no slope applied to the hazard on enrollment date). So this gives you the adjusted hazard over time from enrollment to the end of data.

Apply h_adj(t)=h(t)·e^{−β·t} from enrollment forward where t is the week number (week 0 is enrollment). No centering, no scaling. These step are done elsewhere. Note that this is ONLY a spec for the derivation of the beta and applying it to create adj hazard. nothing else.

12 week periods, a year apart, both in quiet zones should give the most accurate slope normalization. 

W1=[2022-24, 2022-36]
W2=[2023-24, 2023-36]
W3=[2024-12, 2024-20]

For enrollment dates after W1 start, use W2 and W3 as the weeks, i.e., use the next two available windows that are at least 6 months post enrollment.

Inclusive of endpoints. Use that for ALL computations. If there is no hazard data on a week available, just assume it is zero.

These windows are global, for all cohorts.

If a cohort lacks data in the window, use a hazard value of 0 for the beta computation.
If Wm1 or Wm2 is zero, skip slope normalization for the cohort and set beta to 0.

Compute (unadjusted) hazards as usual. this is only slope normalization to derive the hazards_adj value for each dose cohort.

the slope normalization is computed for each YoB (except ASMR), Enrollment, and dose number combination. This is because each is a different cohort of people in the group.

## Other
Keep DYNAMIC_HVE_SKIP_WEEKS as-is; it only affects accumulation, not β estimation or applying the scaling.
Remove all SIN code/params and related debug, keep β printed in summary?

## Output
There is no beta for the ASMR line so don't print the beta values for the ASMR row

Remove slope/scale columns in dose_pairs/by_dose. Leave rest of reporting as is. If there are any unused columns in the output of any of the sheets, remove them (e.g., MR_adjusted, etc from old algorithms)

Remove all SIN artifacts; only keep β per (YoB,Dose) printed in KCOR summary (beta_num/beta_den). No scale_factor/slope columns anywhere. Also drop CH_actual_* and MR_adj_* everywhere.

## Docs
Update README to slope2 with those windows and origin-based application.
Replace SIN section with slope2 description; note the fixed W1/W2 and origin-based application.

No debug needed at this point.
