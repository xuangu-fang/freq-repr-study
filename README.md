# Frequency Representation Study

## Goal
Study which representation space makes the frequency trajectory of PDE solution families smoother, lower-dimensional, and easier to interpolate.

## Current Demo Families
1. phase_family
2. screened_poisson

## Current Representations
- raw
- fourier
- amplitude_phase
- pca

## Current Metrics
- local_smoothness
- curvature
- intrinsic_rank
- interpolation_error

## Workflow
1. generate dataset
2. extract representations
3. compute metrics
4. plot figures
5. write summary

## Quick Start
python -m src.main --config configs/experiments/full_grid_small.yaml