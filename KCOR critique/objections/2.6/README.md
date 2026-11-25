# 2.6 Positive-control injection

Evaluation finding N6 from the [November 21, 2026 review](<../../iterative review/evaluations/2/README.md>).

The original equation multiplied the control cohort's baseline hazard by the injection factor. The generator applies the factor to the treated cohort's own weekly hazard among survivors. An individual-level intervention would also change the frailty distribution of subsequent survivors.

Corrected in iteration 2. [SI §S3.1](<../../iterative review/revised/documentation/preprint/supplement.md#s31-construction-of-injected-effects>) now states the multiplier and its reference explicitly. [SI §S4.3](<../../iterative review/revised/documentation/preprint/supplement.md#s43-positive-control-injected-effect>) describes the treated cohort's hazard inside and outside the window. The [generator](../../../test/positive_control/code/generate_positive_control.py) uses the start week and excludes the end week. A fresh LLM evaluation is pending.

[Objection register](<../../iterative review/open-objections.md>) · [All objections](../../README.md)
