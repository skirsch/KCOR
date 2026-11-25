# Eight-week paired sensitivity run

This frozen run repeats the aggregate-block procedure with eight-week blocks on the first 60 datasets of the primary run. Start with the [assessment](../ASSESSMENT.md), [summary](summary.json), [data and recurrence audit](audit.json), [optimization audit](optimization_audit.json), and [manifest](manifest.json). Numbered JSON/NPZ pairs retain dataset results and bootstrap draws. The dataset counts and point estimates are paired with the same numbered records in `../main/`.

The [datasets.zip](datasets.zip) archive holds the generated evidence for this run. To restore the original paths for the analysis scripts, run `python -m zipfile -e datasets.zip .` from this directory. See the [archive index](../../../../EVIDENCE_ARCHIVES.md).
