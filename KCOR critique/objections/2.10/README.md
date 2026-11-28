# 2.10 Selection of empirical examples

Evaluation finding N10 from the [November 21, 2026 review](<../../iterative review/evaluations/2/README.md>).

The illustration selects on the final KCOR and on improvement in curvature. A Czech configuration comment cites observed harm in motivating window choice. The filter is verifiable; the comment alone cannot establish when the windows were chosen.

[Illustration selector](../../../code/generate_kcor_empirical_intuition_figure.py) · [Czech configuration](../../../data/Czech/Czech.yaml)

[Objection register](<../../iterative review/open-objections.md>) · [All objections](../../README.md)

## Figure 1 selection

The original lead-in and caption called the comparison representative. In iteration 2 the lead-in discloses the final KCOR range and the preference for reduced curvature. The caption now calls the selected comparison an analysis example. The figure and selector are unchanged. A fresh LLM evaluation is pending. The script filters grouped 24-month comparisons to a final KCOR between 0.85 and 1.15, a maximum at most 1.5 and a positive minimum. It gives priority to comparisons where normalization reduces the curvature gap, then ranks eligible comparisons first by the smaller cohort's size. The selected comparison ended at 0.89487 and its curvature gap decreased. [Selector](../../../code/generate_kcor_empirical_intuition_figure.py#L101-L146) · [Figure 1 selection record](../../../documentation/preprint/figures/fig_kcor_empirical_intuition.json)

## Czech window selection

The [January 25, 2026 commit](https://github.com/skirsch/KCOR/commit/0ea60f252600b3ce673f2be8e2b6a44a90ade894) activated the full 2023 calendar year in place of 2022-W24 through 2024-W16 and added the comment citing the observed result. The [March 19 commit](https://github.com/skirsch/KCOR/commit/b5a189d9dacee43b31aba3aa60a4d9484ed0a267) changed the 2023 interval in the v7 multiwindow configuration from weeks 22–37 to weeks 19–40. The January rationale is documented for the full-year choice. Its applicability to the later boundaries remains unverified. In iteration 2, [§2.12](<../../iterative review/revised/documentation/preprint/paper.md#212-worked-example-descriptive>) describes the worked example as fitting across “configured quiet windows.” Prespecification remains part of the general method. SI §S6.1.3 refers separately to the assessment period and to the settings for fitting frailty. A fresh LLM evaluation is pending.
