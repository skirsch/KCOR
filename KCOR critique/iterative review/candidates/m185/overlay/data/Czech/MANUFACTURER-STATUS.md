# Obsolete manufacturer workbooks (15.1)

The three manufacturer result tabs in KCOR.xlsx and three manufacturer input tabs in KCOR_CMR.xlsx are unavailable in the active m174 release pending a corrected full-source regeneration. Each tab retains its original name and now displays an explicit obsolete/unavailable notice, reason, and archive reference. The old numerical estimates and source counts are not exposed as current results; result and interval cells are blank. All six tabs have a red tab color.

The exact original full workbooks are preserved in persistent checkpoint m173. Original hashes match the frozen round-15 input manifest. All unrelated workbook package entries were independently compared byte for byte; none changed. Both edited files reopen successfully, and the three obsolete source sheets are rejected by the current manufacturer reconciliation function. The corrected producer and converter and all paper artifacts are unchanged. No new manufacturer analysis was performed.

## KCOR.xlsx

- Historical original: `work/kcor-review-candidates/m173/data/Czech/KCOR.xlsx`
- Active version: `work/kcor-review-candidates/m174/data/Czech/KCOR.xlsx`
- Original SHA-256: `21f3b2a60dd0efc7bdcd602db810cdbd41b54dc46913c82a7de80629da9c36a3`
- Active SHA-256: `d2fbba56deb055dd4f0da72d87b667d84a3b4e18908010139713ca475c9439f5`
- 13 unrelated OOXML parts identical.
- Affected tabs: MFG_MP_2021_24_D2_decades, MFG_MP_2022_06_D3_decades, MFG_MP_2022_47_D4_decades.

## KCOR_CMR.xlsx

- Historical original: `work/kcor-review-candidates/m173/data/Czech/KCOR_CMR.xlsx`
- Active version: `work/kcor-review-candidates/m174/data/Czech/KCOR_CMR.xlsx`
- Original SHA-256: `b736e53c4d84d709c68ff364f2a415483e3d26b35669debd8bdae3f82c7a6c9c`
- Active SHA-256: `2c456001f2bc05460e414bb3734580ce6bf39acc10f624828638db01c0d73317`
- 23 unrelated OOXML parts identical.
- Affected tabs: 2022_47_MFG_D4, 2022_06_MFG_D3, 2021_24_MFG_D2.


Regen: run the corrected fixed-enrollment converter on the original registry source and the ordinary KCOR reporting pipeline. Do not use the archived manufacturer-only counts as valid input.
