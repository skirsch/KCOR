# Objection 6, first item: weekly inversion

The manuscript's gamma-frailty identity gives an exact cumulative inversion:

\[H_0(t)=\frac{e^{\theta H_{\mathrm{obs}}(t)}-1}{\theta}.\]

For an observed **integrated weekly hazard increment** \(h_j=H_{\mathrm{obs}}(j+1)-H_{\mathrm{obs}}(j)\), subtracting the inversion at consecutive weeks gives

\[H_{0,j+1}-H_{0,j}=(1+\theta H_{0,j})\frac{e^{\theta h_j}-1}{\theta}.\]

The manuscript's Step 2 update instead uses \((1+\theta H_{0,j})h_j\). This is its first-order approximation, because \((e^{\theta h_j}-1)/\theta=h_j+\theta h_j^2/2+\cdots\). When \(\theta h_j>0\), the written update underestimates the exact weekly increment. The distinction applies to the weekly integrated hazard actually used by the code, \(-\log(1-D_j/Y_j)\). If the paper intends an Euler step for a continuous-time differential equation, it should call it an approximation and specify the time-step error.

The code contains **both** paths. `invert_gamma_frailty` in `code/KCOR.py` computes the exact cumulative formula using `expm1`; final KCOR normalization calls this function. Inside `fit_theta0_gompertz`, an internal Step 2 reconstruction uses the approximate recurrence while estimating persistent offsets between quiet windows. The one-window result of objection 3 should not be attributed to that internal recurrence without an additional calculation: the reconstruction has no gap offset to feed when there are no gaps, and other fitting approximations were already identified. The multi-window effect is measured in a separate [isolated check](../objection6_weekly_inversion_multiwindow/README.md).

The [independent arithmetic check](check.py) uses the model-consistent mortality scale from the objection 3 scenario, with \(k=.001\), \(\gamma=.085/52\), two skipped weeks, and frailty 4 or 0.5. Its [retained results](results.json) show a maximum single-week relative underestimation of about 0.198% for frailty 4 and 0.0278% for frailty 0.5. The accumulated week-52 difference is about 0.206% and 0.0260%, respectively. These numbers compare recurrence formulas at known parameters; they do **not** measure the change in fitted parameters, final KCOR, or interval coverage.

Current disposition: the manuscript's characterization of the weekly Step 2 update as the exact discrete inversion is mathematically incorrect. The production final inversion is exact in the tested numerical regime. A manuscript correction can state that Step 2 uses a first-order update, give the exact weekly formula, and distinguish it from the exact final cumulative inversion. The separate multi-window check found an approximation-induced diagnostic switch on a model-generated case; no production change follows from this arithmetic check alone.
