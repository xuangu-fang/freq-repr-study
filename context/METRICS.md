# Metrics

## local_smoothness
S(delta) = mean ||r(w+delta) - r(w)|| / |delta|

## curvature
C(w_i) = ||r(w_{i+1}) - 2 r(w_i) + r(w_{i-1})||

## intrinsic_rank
Use PCA explained variance, effective rank, or participation ratio.

## interpolation_error
Hide intermediate frequencies, interpolate in representation space, reconstruct if possible, compare against ground truth.

## Reporting
All metrics should be reported:
- per trajectory
- aggregated by demo family
- mean and std