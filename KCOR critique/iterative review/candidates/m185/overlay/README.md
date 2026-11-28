KCOR

The combined manuscript, its Markdown sources and figures are in documentation/preprint. Method code is in code and scientific inputs and analyses are in the corresponding repo directories.

### Inclusive population totals

The ordinary disjoint layout defaults to `KCOR_AGGREGATE_SCOPE=exhaustive`: supplied inclusive totals must equal the corresponding disjoint weekly counts. Use `KCOR_AGGREGATE_SCOPE=partial` only for an explicitly incomplete age extract with an independently supplied valid wider total. The wider point can be reported, but its joint bootstrap interval is unavailable when constituent strata are absent. Unrelated supported age-group intervals remain available. The declared overlapping Table 8 layout is unchanged.

In partial scope, a reconstructed aggregate from available disjoint age groups is labeled **All available ages** (population_label), not All Ages. An inclusive total and a partial subset are not mixed across doses. In exhaustive scope, a missing dose-specific total is reconstructed from the complete disjoint components.

Supplied all-ages and born-1960-or-later totals are checked jointly for feasible weekly complementary population counts and death histories, including displayed older and unknown-birth-year strata. Incompatible aggregates are withheld without invalidating independent age groups.

For All available ages, the workbook and log list the supplied birth-year group labels separately for each dose, including explicitly recorded zero-enrollment groups and whether the supplied age-group sets differ. An omitted group is not treated as a documented zero. Bootstrap draws must retain the same reported component coverage.
