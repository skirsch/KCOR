
### see the full chat
https://chatgpt.com/share/68f45b7d-a9e0-8009-89ea-b1c4fa46ab41


### HVE tests

Here are three discriminators that do help tease them apart, and I’ve packaged them into a new script you can run now:

Calendar-symmetric masking (A/B test)

A: Mask the first K calendar weeks only for the vaccinated (what we did before).

B: Mask the same K calendar weeks for both vaccinated and unvaccinated.

If the harm signal collapses only under B (but not under A), that points to anchor/early-window bias. If it stays elevated under B, that’s consistent with true exposure-linked excess risk.

Pseudo-exposure null

Mask K random calendar weeks (same K as above) for both groups, repeat R times to get a null distribution for endpoint KCOR change.

If the observed change from A (or B) sits well outside this null, that supports an exposure-tied effect rather than an arbitrary early-window artifact.

Endpoint check at 26/52/78 weeks

Report how KCOR (or the KCOR-like ratio) moves under A, B, and vs the pseudo-null. Direction and magnitude matter.

I’ve put all three into this script:

Download: sandbox:/mnt/data/hve_discriminator.py

Example runs (80–89yo = YOB 1932–1941):

python hve_discriminator.py --enroll 2021_24 --yob 1932 1941 --num 3 --den 0 --k 3 --weeks 26 --R 500

python hve_discriminator.py --enroll 2022_06 --yob 1932 1941 --num 3 --den 0 --k 3 --weeks 26 --R 500

Outputs (in ./out):

hve_disc_series_<...>.csv time series for A (asym), B (sym), and one random C example

hve_disc_endpoints_<...>.csv endpoints for A and B at ~26/52/78

hve_disc_endpoints_null_<...>.csv null distribution from the pseudo-masking

How to read it quickly

If A drops toward 1.0 but B stays ≈ original, that suggests true early harm (the asymmetric mask removes early harmful weeks only in vax).

If both A and B drop similarly (and align with the null), early-window/anchor bias is the driver.

If neither drops (and they lie outside the null), the harm signal is robust to these controls.
