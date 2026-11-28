# Why grouped person resampling is still person resampling

In this one-stratum simulation, each original person belongs to exactly one category: cohort 0 or 1, and a death week or survival through follow-up. Let `C_j` be the original number of people in category `j` and `N` the total number of people. Sampling `N` original people independently **with replacement** chooses category `j` with probability `C_j/N`. The resulting vector of category counts is therefore exactly

`Multinomial(N, C_1/N, C_2/N, ...)`.

The [reused simulation function](../../objection3_corrected_validation/simulation.py) draws this multinomial vector and rebuilds weekly deaths and risk sets. It is a storage-efficient representation of the software's record-level person bootstrap, not independent weekly death-count noise. The retained [aggregation verification](../../objection3_resolution/verify_person_aggregation.py) executed the original person-record aggregation and matched the grouped count representation.

This equivalence applies to the declared single-stratum category data. If an applied analysis stratifies or weights people differently, the category representation must preserve those distinctions before the same argument applies.
