# COVID adjustment

This is a simple h(t) scaling adjustment applied only to excess hazards during a COVID interval.

The parameters for the correction are determined in the config.yaml file for the dataset.

Set these Parameters for the Czech dataset yaml file only:
- COVID_CORRECTION_START_DATE: "2021-41"
- COVID_CORRECTION_END_DATE: "2021-52"
- COVID_CORRECTION_FACTOR: 1.4

If the parameters aren't specified, the correction is not done.

The correction is applied ONLY to the unvaccinated cohort.

The correction is made immediately AFTER h(t) is computed from the data and before it is cumulated to produce H(t).

So H(t) is computed from the adjusted data.

The correction is simple.

Look at h(t) at the COVID_CORRECTION_START_DATE. call this h0

Apply the adjustment formula from after the start date to the END_DATE inclusive:
h'(t) = (h(t)-h0)/COVID_CORRECTION_FACTOR + h0

That's it.

Bump the revsision number to 6.1 and mention this in the code and the README file.
