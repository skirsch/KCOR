1) Cumulative hazard notation (observed vs baseline vs depletion-neutralized)
1.1 Disallowed observational superscripts: H_d^{\mathrm{obs}}(t) / h_d^{\mathrm{obs}}(t)
Locations (examples; repeated throughout):
Box 1 step 2/3/4: L144, L146, L148 in paper.md
Notation table: L162–L163
Methods §2.3: Eq @eq:hazard-discrete L292; Eq @eq:cumhazard-observed L299
Methods §2.4.2: Eq @eq:gamma-frailty-identity L339
Multiple later references: e.g., L380, L392, L397, L405, L453, L471, L476, L505–L506, L864, L1181
Current form (representative snippets):
$H_d^{\mathrm{obs}}(t)$, $h_d^{\mathrm{obs}}(t)$
Violation type:
Observational status must not be a superscript; cohort index must be explicit and stable.
Canonical target form:
Use subscripts for observational status and keep cohort index explicit:
H_{\mathrm{obs},d}(t) and h_{\mathrm{obs},d}(t) (by direct analogy with the cumulative-hazard rule).
Minimal mechanical fix:
Replace every H_d^{\mathrm{obs}}(t) → H_{\mathrm{obs},d}(t)
Replace every h_d^{\mathrm{obs}}(t) → h_{\mathrm{obs},d}(t)
Apply consistently in text, tables, equations, captions, and appendices.
1.2 Missing cohort index in hazards/cumhazards: H^{\mathrm{obs}}(t) / H_0(t) / h_0(t)
Locations:
Figure caption for @fig:kcor_v6_schematic: L352
Appendix A.1–A.3 (multiple): L954–L1006
Current form:
$H^{\mathrm{obs}}(t)$, $H_0(t)$, $h_0(t)$
Violation type:
Cohort index must be explicit everywhere (no “context carries the index”), and notation must be isomorphic across Methods/figures/appendix.
Canonical target form:
H_{\mathrm{obs},d}(t) and H_{0,d}(t) (and corresponding hazards if used).
Minimal mechanical fix:
In captions/appendix derivations, introduce d and rewrite:
H^{\mathrm{obs}}(t) → H_{\mathrm{obs},d}(t)
H_0(t) → H_{0,d}(t)
h_0(t) → h_{0,d}(t)
Preserve meaning: it’s the same derivation, just indexed.
1.3 Depletion-neutralized cumulative hazard notation is non-canonical: \tilde H_{0,d}(t) (missing braces)
Locations (examples):
Box 1 step 4/5: L148–L150
Eq @eq:normalized-cumhazard: L421–L423
Eq @eq:normalized-hazard-diff: L432–L434
Many prose references: L437, L439, L451, L650, Appendix A.4 L1018
Current form:
\tilde H_{0,d}(t) and \tilde h_{0,d}(t) (tilde applied to a space-separated symbol)
Violation type:
Canonical depletion-neutralized object is \tilde{H}_{0,d}(t) (stable symbol set; typographic variation is disallowed).
Canonical target form:
\tilde{H}_{0,d}(t) (and \tilde{h}_{0,d}(t) if used)
Minimal mechanical fix:
Replace \tilde H_{...} → \tilde{H}_{...}
Replace \tilde h_{...} → \tilde{h}_{...}
2) Gamma-frailty identity + inversion must match the canonical typography
2.1 Main-text identity uses the wrong observed cumulative hazard symbol
Location:
Eq @eq:gamma-frailty-identity: L338–L340
Current form:
H_d^{\mathrm{obs}}(t) = (1/\theta_d)\log!(1+\theta_d H_{0,d}(t))
Violation type:
Observed cumulative hazard must be H_{\mathrm{obs},d}(t) (not H_d^{\mathrm{obs}}(t)).
Canonical target form:
As above, with H_{\mathrm{obs},d}(t) on the LHS.
Minimal mechanical fix:
Change only the observed hazard symbol on LHS (and any matching RHS occurrences elsewhere).
2.2 Inversion uses e^{...} instead of canonical \exp\!(...)
Locations (examples):
Eq @eq:gamma-frailty-inversion: L345–L347 uses e^{...}
Eq @eq:normalized-cumhazard: L421–L423 uses e^{...}
Appendix A.3: L1001–L1006 uses e^{...}
Box 1 step 4: L148 uses e^{...}
Current form:
\frac{e^{\theta_d H_d^{\mathrm{obs}}(t)} - 1}{\theta_d}
\frac{e^{\hat \theta_d H_d^{\mathrm{obs}}(t)} - 1}{\hat \theta_d}
Violation type:
Canonical inversion is fixed: must use \exp\!\left(\cdot\right) and canonical observed hazard symbol.
Canonical target form:
\frac{\exp\!\left(\theta_d\,H_{\mathrm{obs},d}(t)\right)-1}{\theta_d}
Minimal mechanical fix:
Replace every e^{X} inversion form with \exp\!\left(X\right) and update the inner symbol to H_{\mathrm{obs},d}(t) (or the hatted version only where it is explicitly a fitted quantity).
2.3 Box 1 uses a non-canonical identity specialization and inconsistent \log typography
Location:
Box 1 step 3: L146
Current form:
$H_d^{\mathrm{obs}}(t) = \frac{1}{\theta_d}\log(1+\theta_d k_d t)$
Violation type:
Canonical identity must be visually identical everywhere; also observed hazard symbol non-canonical.
Canonical target form:
Use canonical identity and (if needed) separately state the constant-baseline assumption as H_{0,d}(t)=k_d\,t (see §3).
Minimal mechanical fix:
Keep the constant-baseline assumption, but make the displayed identity itself match the canonical form (same \log\!\left(\right) structure; same symbols).
3) Quiet-window baseline assumptions (no auxiliary baseline-shape symbols)
3.1 Auxiliary baseline shape g(t) is introduced even though baseline is declared constant
Location:
Eq @eq:baseline-shape-default: L358–L364
Current form:
h_{0,d}(t)=k_d\,g(t), g(t)=1, H_{0,d}(t)=k_d t
Violation type:
If the baseline is constant, the assumption must be stated directly; do not introduce auxiliary baseline-shape symbols unless used later.
Canonical target form:
H_{0,d}(t)=k_d\,t (and if needed h_{0,d}(t)=k_d)
Minimal mechanical fix:
Remove g(t) entirely and keep only the constant-baseline form in canonical typography.
3.2 Constant-baseline typography deviates from the canonical simplest form
Location:
Same block L363: H_{0,d}(t)=k_d t
Violation type:
Canonical example includes spacing: k_d\,t.
Minimal mechanical fix:
k_d t → k_d\,t (and keep cohort index).
4) Parameters vs estimates (hats discipline + cohort subscripts never dropped)
4.1 Dropped cohort subscript on fitted frailty parameter: \hat{\theta} used as if it were a cohort-specific estimate
Locations:
Prose: L252 uses $\hat{\theta} \approx 0$
Table 2 header/caption/bullets: L595, L605, L608–L616, L620, and more prose later (e.g., L642)
Current form:
\hat{\theta} without _d
Violation type:
Do not drop the cohort subscript on any parameter or estimate.
Canonical target form:
Fitted estimates must be hatted and indexed: \hat{\theta}_d (and similarly \hat{k}_d).
Minimal mechanical fix:
Replace \hat{\theta} → \hat{\theta}_d (or an explicitly defined index if the table aggregates multiple cohorts; but per rules, index must not be carried only by context).
4.2 Inconsistent hatted forms for the same fitted quantity (\hat\theta_d, \hat \theta_d, \hat{\theta}_d)
Locations (examples):
Box 1 step 4: L148 uses \hat\theta_d
Eq @eq:normalized-cumhazard: L422 uses \hat \theta_d
Appendix / later text uses \hat{\theta}_d (e.g., L1168)
Violation type:
Do not alternate between visually different hatted forms for the same fitted object.
Canonical target form:
\hat{\theta}_d, \hat{k}_d
Minimal mechanical fix:
Standardize all occurrences to \hat{\theta}_d and \hat{k}_d.
5) Estimand vs framework naming (KCOR vs KCOR(t))
5.1 “KCOR” used where the time-indexed estimand is intended (curves/trajectories/values)
Locations (examples):
“KCOR trajectories …” L232, L556, L764, L790, L872, L1278, L1288
“KCOR curves …” L459, L872
“KCOR values …” L706, L751, L753, L1088
Figure captions that describe a curve being flat at 1 without consistently using KCOR(t) (e.g., L574)
Violation type:
KCOR (unqualified) must refer to the framework; the estimand/curve must be KCOR(t).
Canonical target form:
Use KCOR(t) wherever referring to trajectories/curves/values at time t.
Minimal mechanical fix:
Replace prose instances like “KCOR trajectories/curve/values” with “KCOR(t) trajectories/curve/values” when the meaning is the time-indexed estimand.
5.2 Canonical estimand definition typography differs (\mathrm{KCOR}(t) and missing tilde braces)
Locations:
Box 1 step 5: L150
Eq for KCOR: L483
Current form:
\mathrm{KCOR}(t)=\tilde H_{0,A}(t)/\tilde H_{0,B}(t)
Violation type:
Canonical definition is a fixed visual form; also \tilde H should be \tilde{H}.
Canonical target form:
KCOR(t)=\tilde{H}_{0,A}(t)/\tilde{H}_{0,B}(t)
Minimal mechanical fix:
Remove \mathrm{} wrapper and fix tilde bracing.
6) Display-math requirement for composite expressions (tilde/hats/multi-subscripts)
Per your invariant: all composite expressions (tilde, hats, multi-subscripts) must be in display math and display math must not be converted to inline.
6.1 Composite expressions appear inline throughout
Locations (examples):
Box 1 step 2–5: L144–L150 include H_d^{\mathrm{obs}}(t), \tilde H_{0,d}(t), \hat\theta_d, \mathrm{KCOR}(t), \tilde H_{0,A}(t)
Notation table rows: L162–L167 include h_d^{\mathrm{obs}}(t), H_d^{\mathrm{obs}}(t), h_{0,d}(t), H_{0,d}(t), \theta_d, k_d
Prose lines like L451, L437, L439 include \tilde H_{0,d}(t) inline
Many places use \hat{\theta} inline (e.g., L252, L608)
Violation type:
Composite math must be moved into display math blocks.
Minimal mechanical fix (for Phase 2):
Convert each inline composite occurrence into a $$ ... $$ display block without rewriting the expression (and without demoting any existing display math).
Next step
This completes Phase 1 (violation discovery + canonical targets). If you approve, I’ll proceed to Phase 2 and apply only the minimal mechanical edits to paper.md to enforce these invariants.
Approve Phase 2? (yes/no)