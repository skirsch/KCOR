Here is a **clean, exact, drop-in instruction you can give to Cursor**, written as an imperative edit task. You can paste this verbatim.

---

### **Cursor instruction — Appendix E table conversion**

**Task:** Convert Appendix E (“Reference Implementation and Default Settings”) from bullet prose into a single summary table, while preserving the introductory prose.

**Steps:**

1. **Keep the opening two paragraphs of Appendix E unchanged**, beginning with

   > “This appendix documents the reference implementation and default operational settings…”

2. **Remove subsections E.1, E.2, and E.4 and their bullet lists entirely.**

3. **Insert a single table** in the Tables section of the document after # Appendix E with the following Markdown:

```markdown
| Component | Setting | Default value | Notes |
|---|---|---|---|
| Cohort construction | Cohort indexing | Enrollment period × YearOfBirth group × Dose; plus all-ages cohort (YearOfBirth = −2) | Implementation detail |
| Quiet-period selection | Quiet window | ISO weeks 2023-01 through 2023-52 | Calendar year 2023 |
| Frailty estimation | Skip weeks | `SKIP_WEEKS = DYNAMIC_HVE_SKIP_WEEKS` | Applied by setting $h_d^{\mathrm{eff}}(t)=0$ for $t < \mathrm{SKIP\_WEEKS}$ |
| Frailty estimation | Fit method | Nonlinear least squares in cumulative-hazard space | Constraints: $k_d>0$, $\theta_d \ge 0$ |
```

4. **Do not add boldface or manual “Table E.X” numbering**; allow Pandoc / LaTeX to assign numbering automatically.

5. **Do not introduce additional prose** beyond the existing introductory paragraphs.

6. Make sure that Appendix E text references the Table explicitly using the cross-referencing feature.

**Goal:** Present Appendix E as a concise reference table that documents the prespecified default operational settings used for reproducibility.

---

