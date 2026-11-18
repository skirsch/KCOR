# Verified Monte Carlo implementation audit

This checks the existing production interval path and selected saved result rows. The tested production path resamples people, rebuilds weekly counts, reruns KCOR, and uses empirical percentiles. That fact alone does not establish interval coverage. Read [the assessment](assessment.txt), [verification](verification.json), and [the main objection](../README.md).
