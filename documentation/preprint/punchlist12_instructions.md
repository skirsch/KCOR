TASK: Enforce notational invariants on the manuscript.

CONTEXT:
The file `punchlist12_jlw_feedback.md` defines hard, non-negotiable notation and estimand rules.
Treat these as invariants, not suggestions.

SCOPE:
- Apply these invariants mechanically to the manuscript.
- Do not change scientific meaning.
- Do not introduce new symbols.
- Do not simplify notation.
- Do not inline composite math expressions.
- Do not alter prose except where required to enforce invariants.

REQUIREMENTS:
1. Make all cumulative hazard notation conform exactly to the canonical forms.
2. Ensure cohort indices, hats, and tildes are preserved everywhere.
3. Ensure parameters vs estimates are never conflated.
4. Ensure KCOR refers only to the framework and KCOR(t) to the estimand.
5. Make figures, captions, appendices, and Methods notationally isomorphic.
6. Use display math for all composite expressions (tilde, hats, multi-subscripts).
7. Do not reformat math unless required to satisfy these rules.
8. Do not convert display math into inline math under any circumstances.


PROCESS:
- First, produce a build plan listing all violations found.
- Do NOT modify the manuscript yet.
- After approval, apply fixes in a second pass.

OUTPUT:
- A structured list of violations with file locations.
- No manuscript edits in this step.
