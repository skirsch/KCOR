# Manufacturer-only corrections (14.1 and 14.2)

Persistent checkpoint: m171, parent m170. No manuscript or core estimator change.

Original intent: Git 4fe94be8 and 7be6f1b6, October 21, 2025, introduced these M/P KCOR outputs by dose and decade. They are not part of the paper or supplement.

- 14.1: Original intent verified in Git commits 4fe94be8 and 7be6f1b6 (October 21, 2025): explicitly labeled M/P KCOR by dose and decade. Replaced duplicate raw-hazard implementations with a shared ordinary KCOR calculation, using the actual enrollment date, usual stabilization skip, all-window frailty fit, strict anchoring, and the ordinary person bootstrap. Workbook and log use the same results. The old incompatible source is reported unavailable rather than producing a misleading number.
- 14.2: Manufacturer histories now partition the actual ordinary converter enrollment-dose assignment, including the other/unknown category. Weeks start at enrollment and opening counts are reduced only by deaths in that fixed cohort. Manufacturer sums must reconcile with ordinary weekly birth-year/sex/DCCI counts. The test includes a person already on the next dose who is excluded, and an unknown manufacturer retained in the other category.


## Verification

- The fixed-enrollment assignment test excludes an individual already on the next dose and retains an unknown manufacturer. Count-reconciliation failure is rejected.
- Integrated 85-week synthetic workbook, including pre-enrollment rows: 1940s endpoint 1.000008 [0.837751, 1.322868] with 3/3 valid draws; fitted variances 0.994944 and 1.131169. Workbook and log agree at the configured report date (1.0000 [0.7994, 1.2457]). Three draws test code/availability only.
- Actual CMR converter executed on the 1,811-record birth-band subset of the existing 9,999-record Czech sample. The 2021-24 dose-2 opening risk sets were M=109 and P=647. Every stratum/week reconciled with ordinary output, with valid death-only histories.
- Released full Czech workbook 2022-06 dose-3 manufacturer's histories failed the new reconciliation check as expected; not silently analyzed. Full source regeneration and 999-draw manufacturer analyses were not performed. The historical workbook is preserved.
- Seventeen checked manuscript, bootstrap and main numerical archive files match m170 byte-for-byte. Paper PDFs unchanged.

Remaining round-14 objections: 14.3, 14.4.
