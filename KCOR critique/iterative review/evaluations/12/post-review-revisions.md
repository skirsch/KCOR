# Round 12 post-review revisions

Frozen m161 preserved; m162 approved straightforward batch.

- [12.1](../../../objections/12.1/README.md): Removed obsolete calendar cutoff; retain configured age display range and MC range. Valid 2023 enrollment with post-2020 birth group gives identical all-ages point and all three matched bootstrap intervals in both input representations: 1.981297 [1.794494, 2.140051] (three-draw smoke only). Published Czech input eligibility unaffected.
- [12.3](../../../objections/12.3/README.md): Corrected generator axis and rebuilt actual embedded figure from unchanged archived endpoints; visually verified axis reads Observed week-52 KCOR minus one.
- [12.6](../../../objections/12.6/README.md): SI now says complete enumeration introduces no extra selective subsampling, not that depletion is small or frailty poorly identified.
- [12.9](../../../objections/12.9/README.md): Pooled frailty parameter status cells blank; separate contributing-fit flags retained; written workbook parity regression verifies both status cells empty.
- [12.10](../../../objections/12.10/README.md): Cox figure caption distinguishes systematic differential frailty depletion from sampling variation of individual estimates. No numerical changes.
- [12.11](../../../objections/12.11/README.md): Historical S7 first-week nominal population mismatch and two-week skip explicitly documented; original code and archive unchanged; any future rerun must use consistent initial risk and mixture mass.

Remaining for individual discussion: 12.2, 12.4, 12.5, 12.7, 12.8. No change to frailty, weighting or wave inversion; published primary Table 8/S6 estimates and intervals not altered. Earlier forced-alpha full intervals remain archived.

- m163: 12.2 weekly identity attribution and diagnostic counts corrected; cumulative output unchanged. Controlled fixtures and nonlinear continuity regression pass. Remaining 12.4, 12.5, 12.7, 12.8.

- m164: 12.4 diagnostic-only signed seed. Exact protective-gap and 50-draw regression resolves false later deficit; 999-draw primary Table 8 points, draws and parameters identical, all gap status classifications unchanged. Remaining 12.5, 12.7, 12.8.

- m165: 12.5 historical relaxation figure attribution corrected in main and SI, original figure unchanged. Remaining 12.7, 12.8.

- m166: 12.7 paper-specific Table 8 replay uses same available-strata and failure rules, retains contributor masks, and withholds intervals if draw population changes. Targeted ordinary/replay parity and 999-draw primary exact replay passed; general KCOR code and manuscript unchanged. Remaining 12.8.

- m167: 12.8 common finite-positive reference support for direct, pooled and paper replay; numerical stress and existing composition regressions pass. Full primary 999-draw replay agrees within floating-point precision, same intervals and diagnostics. All eleven round-12 findings have implemented responses; independent closure pending.
