# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a controlled representation study for frequency trajectories of PDE solution families. The goal is to determine which representation space makes the frequency trajectory smoother, lower-dimensional, and easier to interpolate/extrapolate.

Key concepts:
- **Demo families**: synthetic PDE solution families (phase_family, screened_poisson, helmholtz, wave_equation)
- **Frequency trajectory**: a set of fields {u_w1, u_w2, ..., u_wM} for a fixed parameter sweep
- **Representations**: raw, fourier, amplitude_phase, pca, real_imag, autoencoder, amplitude_phase_autoencoder, enhanced_real_imag_autoencoder
- **Metrics**: local_smoothness, curvature, intrinsic_rank, interpolation_error

## Common Commands

### Run an experiment
```bash
python -m src.main --config configs/experiments/full_grid_small.yaml
```

### Generate trajectory GIFs (for complex fields)
```bash
python -m src.report.complex_trajectory_gif \
  --results outputs/experiment_name \
  --demo demo_name \
  --output outputs/gifs \
  --fps 3 \
  --multipanel
```

### Install dependencies
```bash
pip install -r requirements.txt
```

## Architecture

The codebase follows a modular, config‑driven design:

```
src/
├── data_gen/          # Generate demo‑family trajectories
├── representations/   # Transform trajectories into different representation spaces
├── metrics/          # Compute smoothness, curvature, rank, interpolation error
├── experiments/      # Orchestrate benchmark runs
├── report/           # Generate figures and summaries
└── utils/            # Shared utilities
```

### Representation Modules

Each representation module implements:
- `fit(train_data, config)` (if needed)
- `transform(trajectory)`
- `inverse_transform(...)` (if available)
- `metadata()`

Available representations:

1. **Raw**: Field flattened as baseline
2. **Fourier**: FFT coefficients
3. **Amplitude-phase**: Hilbert transform extracts amplitude and phase (supports phase unwrapping)
4. **PCA**: Linear dimensionality reduction (95% variance)
5. **Real-imag**: Complex field decomposed into real and imaginary parts
6. **Autoencoder**: Convolutional autoencoder learns nonlinear representation (real-imag input)
7. **Amplitude-phase autoencoder**: Autoencoder with amplitude and phase as input channels
8. **Enhanced real-imag autoencoder**: Deeper autoencoder with residual connections (real-imag input)

Complex field support:
- Raw, Fourier, Amplitude-phase, Real-imag: Native complex support
- Autoencoder variants: Support complex fields via real/imag or amplitude/phase channels

### Configuration
Experiments are defined by YAML configs in `configs/experiments/`. Each config specifies:
- `dataset`: number of trajectories, train/val/test split
- `demos`: list of demo configs (from `configs/demos/`)
- `representations`: which representation modules to apply
- `metrics`: which metrics to compute
- `output_dir`: where to save results

Demo configs (in `configs/demos/`) define the parametric family and its parameter grid.

### Representation modules
Each representation must implement:
- `fit(train_data, config)` (if needed)
- `transform(trajectory)`
- `inverse_transform(...)` (if available)
- `metadata()`

### Output structure
Results are saved under `outputs/` with subdirectories:
- `datasets/`       # Raw trajectory data
- `representations/` # Cached representation transforms
- `metrics/`        # Metric tables (per trajectory, aggregated)
- `figures/`        # Visualizations
- `tables/`         # Formatted result tables
- `summaries/`      # Markdown summaries

Each run saves a config snapshot and a manifest of generated files.

## Development Principles

1. **Respect the context files** – Read `context/*.md` before implementing new features.
2. **Config‑driven changes** – Avoid hard‑coding parameters; add config options instead.
3. **Modular, reusable code** – Prefer extending existing modules over one‑off scripts.
4. **Reproducible outputs** – Always save config, seed, and file manifest.
5. **Minimal scope** – Do not introduce deep‑learning components or change the research direction without updating the context files.

## Experiment Rules

1. Start with the smallest config (`full_grid_small.yaml`).
2. Do not add new demos or metrics without updating the context files.
3. Save config snapshot, random seed, and output manifest for every run.
4. Use fixed train/val/test splits at the trajectory level.
5. Change only one major factor per benchmark run.

## Workflow

The standard agent workflow (from `context/AGENT_WORKFLOW.md`):
1. **Build dataset** – Generate trajectories for each demo family.
2. **Extract representations** – Transform each trajectory into multiple representation spaces.
3. **Compute metrics** – Evaluate smoothness, curvature, rank, and interpolation error.
4. **Plot and summarize** – Generate figures and markdown summary.
5. **Recommend next step** – Suggest which representation should be used for later generative modeling.

## Notes

- The current scope is 2D fields only, no geometry, no full generative modeling.
- All metrics should be reported per trajectory and aggregated by demo family (mean ± std).
- Chinese may be used for internal summaries (see `agent/CHECKLIST.md`).

## Visualization Commands

After running a benchmark experiment, generate visualization figures:

```bash
# Generate figures for a single experiment
python -m src.report.generate_figures --results outputs/experiment_name

# Compare multiple experiments
python -m src.report.generate_figures --results . \
  --compare outputs/exp1/benchmark_results.pkl outputs/exp2/benchmark_results.pkl \
  --names "Experiment 1" "Experiment 2" \
  --output outputs/comparison_figures
```

The visualization module generates:
1. Representation comparison bar plots
2. Demo comparison bar plots
3. Ranking heatmaps
4. Trajectory visualizations
5. Multi-experiment comparison plots

All figures are saved to `figures/` subdirectories within each experiment output.

### Generate trajectory GIF animations (for complex fields)
For visualizing how fields evolve with frequency, use the complex field GIF generator:

```bash
# Basic usage - generates GIFs for all components
python -m src.report.complex_trajectory_gif \
  --results outputs/experiment_name \
  --demo demo_name \
  --output outputs/gifs \
  --fps 3

# Generate multipanel GIF showing real, imag, amplitude, and phase
python -m src.report.complex_trajectory_gif \
  --results outputs/experiment_name \
  --demo demo_name \
  --output outputs/gifs \
  --fps 3 \
  --multipanel

# Only generate specific components
python -m src.report.complex_trajectory_gif \
  --results outputs/experiment_name \
  --demo demo_name \
  --output outputs/gifs \
  --fps 3 \
  --components real amplitude

# For real fields (not complex), the script automatically adapts
```

The GIF generator creates:
- Individual component GIFs (real, imag, amplitude, phase)
- Multipanel GIFs (2×2 grid for complex fields)
- Point trace GIFs showing spatial point frequency response