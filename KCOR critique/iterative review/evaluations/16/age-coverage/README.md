# 16.1 - Separate available-age coverage by dose

Persistent m179, parent m178. An All available ages contrast now identifies the actual raw supplied birth-year group labels before age bucketing or filtering, separately for numerator and denominator. The per-dose recorded-zero enrollment groups and a same/different supplied age-groups status are also exported. Coverage is visible in ordinary dose-pair and by-dose sheets, summaries/logs and diagnostic outputs. The bootstrap rejects a draw whose recorded scope or coverage differs from its point. No equal-population-composition requirement or numerical change is imposed.

## Verification

The reproduced partial input has numerator groups 1940,1950 and denominator 1940; the status explicitly identifies different supplied age groups. The original m178 endpoint 2.217437 [1.995869, 2.650563] (three draws) is exactly retained in m179. Supplying an explicit zero-enrollment 1950 dose-0 history yields the identical point and interval, but both supplied group lists are now 1940,1950 and the denominator recorded-zero group is 1950. These are smoke-test intervals, not precision studies. Both cases have three valid draws and zero coverage mismatches. The per-dose sheet and log match the contrast metadata.

Actual Czech S6 point replay: all 1,192 rows and 41 prior fields agree exactly with the earlier replay. The 51 checked numerical archive and figure files are byte-identical. No full bootstrap re-run. The paper addition describes the reporting convention; no change to its results.

Remaining: 16.2.
