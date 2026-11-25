# Four-week primary coverage run

This frozen run has 200 independent model-consistent datasets and 199 aggregate-block draws for each. Start with the [assessment](../ASSESSMENT.md), [summary](summary.json), [data and recurrence audit](audit.json), [optimization audit](optimization_audit.json), and [manifest](manifest.json). Numbered JSON files describe each dataset and interval; matching NPZ files retain its bootstrap estimates and fitted frailties. Coverage here applies to the explicit four-week rule in the parent design, not an unspecified historical paper implementation.

The [datasets.zip](datasets.zip) archive holds the generated evidence for this run. To restore the original paths for the analysis scripts, run `python -m zipfile -e datasets.zip .` from this directory. See the [archive index](../../../../EVIDENCE_ARCHIVES.md).
