# Paired person-bootstrap result files

These 200 numbered JSON/NPZ pairs contain the paper joint fit with 199 person-resampling Monte Carlo estimates per dataset. The original dataset count hash and point estimate in each record match the same-numbered record in the [block run](../../block_bootstrap_operationalization/main/README.md). Start with the [assessment](../ASSESSMENT.md), [summary](summary.json), [draw and pairing audit](audit.json), [optimization audit](optimization_audit.json), [paired interval list](paired_intervals.json), and [manifest](manifest.json). The smoke run is a separate two-dataset execution check and is not part of coverage.

The [datasets.zip](datasets.zip) archive holds the generated evidence for this run. To restore the original paths for the analysis scripts, run `python -m zipfile -e datasets.zip .` from this directory. See the [archive index](../../../../EVIDENCE_ARCHIVES.md).
