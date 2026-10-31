# Data

## pl_targets_153.csv
Processed table of 153 path-loss values (9 distances 0.20-1.00 m in 0.10 m steps x 17 angles: 0-15 deg in 1 deg steps plus 20 deg).

| column | meaning |
|---|---|
| row_id | 0-152, distance-major order |
| distance_m | link distance [m] |
| angle_deg | misalignment angle [deg] |
| measured_pl_db | PL = -(arithmetic mean over the 4096 CFR magnitudes in dB, 240-300 GHz) [dB] |
| source_mat_file | original MAT file name in the Karakoca et al. dataset |
| source_mat_sha256 | SHA-256 of that MAT file (as used by the authors) |
| source_sorted_column | column index of the angle in the sorted angle axis of that file |
| target_definition | text definition of the target |

The target retains the fixed 24.8 dBi horn-antenna response and measurement-system effects; it is not de-embedded propagation loss.
The raw files also contain 25 and 30 deg (18 responses); they are outside the modeling domain (theta <= 20 deg) and are not in this table.

## expected_headline_numbers.csv
Values reported in the paper (RMSE in dB; coverage in %), used by `reproduce_validation.py`.

## Provenance and permissions
Source: Karakoca et al., "sub-THz Wireless Channel Measurements", IEEE DataPort, doi:10.21227/txmh-mz22. The source's permission for redistributing derived values has not been independently verified. The release includes only 153 derived scalar values, not the raw files; this distinction alone does not establish redistribution permission. Check the applicable source terms before public release. Raw data: see the links in the top-level README, or contact the authors.
