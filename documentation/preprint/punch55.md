# Cursor punchlist: add agampodi2024 + bakker2025 citations and add Bakker to the HVE table

## Files
- Edit: /mnt/data/paper.md
- Bibliography source (already contains both entries): /mnt/data/KCOR_references.json

---

## 1) Add citations in §1.4

1. Open: /mnt/data/paper.md
2. Find section:
   ### 1.4 Evidence from the literature: residual confounding despite meticulous matching
3. In the paragraph, find the citation bracket:
   [@obel2024; @chemaitelly2025]
4. Replace it with:
   [@obel2024; @chemaitelly2025; @agampodi2024; @bakker2025]

(Do not change any prose.)

---

## 2) Update the HVE summary table at the end

1. Scroll to the end of /mnt/data/paper.md and locate:

Table: Summary of two large matched observational studies showing residual confounding / HVE despite meticulous matching. {#tbl:HVE_motivation}

2. Change "two" -> "three" in the caption:

Table: Summary of three large matched observational studies showing residual confounding / HVE despite meticulous matching. {#tbl:HVE_motivation}

3. In the table body, add a new row for Bakker immediately after Chemaitelly (keep the same column structure):

| Bakker et al. (Netherlands) [@bakker2025] | National observational mortality + COVID-19 vaccination cohort | Stratification + matching on multiple population and health indicators; survival analyses + calendar-time sanity checks | Evidence consistent with residual HVE despite matching; additional evidence of vaccination-status misclassification artifacts | Demonstrates how strong selection + misclassification artifacts can dominate VE/safety estimates; underscores need for diagnostics-first cohort comparison |

---

