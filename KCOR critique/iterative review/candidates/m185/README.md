# Current review candidate m185

Frozen candidate evaluated in round 21 (28 November 2026). It is the hypothetical integrated edition, not approval or replacement of the production root. The overlay contains all files whose frozen bytes differ from the existing PR root, except the two raw registry sources already present in the root as Git LFS objects. The aggregate CMR workbook uses Git LFS; the raw registry is not duplicated. The exact 1,387-file SHA-256 manifest is included. The earlier baseline and checkpoints remain unchanged.

## Select or reproduce this edition

From the PR checkout, fetch the LFS objects (`git lfs pull`), then run:

```bash
python "KCOR critique/iterative review/candidates/m185/materialize.py" ../KCOR-m185
```

This validates every frozen file before copying and refuses to overwrite an existing directory. Use `--verify` to check without copying; `--source-root` can point to another checkout containing identical base files. Run the usual code and tests from the new directory. No Git rollback or production-root modification is required. The candidate contains the final clean manuscript, and the cumulative linked redline source is adjacent to this README. To rebuild that redline, run XeLaTeX twice on `redline.tex` from the materialized `documentation/preprint` directory (use an absolute path to the redline source).

## Review

[Current state](../../../CURRENT-STATE.md) | [Round 21](../../evaluations/21/convergence.md) | [Clean PDF](../../KCOR-clean.pdf) | [Linked redline](../../KCOR-redline.pdf). Per-objection intermediate PDF replicas are retained locally, not duplicated in this PR; the current pair and previously packaged checkpoints are included. Restored per-draw files in the earlier compressed evidence archives need not be duplicated. The omitted local paths are listed in `excluded-local-artifacts.json`.
