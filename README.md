# Frequency Representation Study

## Overview
This project systematically studies the smoothness, low-dimensionality, and interpolation performance of frequency trajectories for PDE solution families across different representation spaces. By comparing traditional representations with learning-based representations, we explore optimal representation strategies for complex-valued PDE frequency trajectories.

## Key Research Questions & Answers

### Core Research Questions (Q1-Q4)
1. **Q1: In which representation space is the frequency trajectory smoother?**
   - **Answer**: Learning-based representations (autoencoders) are smoothest; among traditional methods, PCA is optimal

2. **Q2: In which representation space is the frequency trajectory lower-dimensional?**
   - **Answer**: Learning-based representations learn compact representations; among traditional methods, PCA has lowest dimensionality

3. **Q3: Which representation space yields better interpolation of missing frequencies?**
   - **Answer**: Learning-based representations have smallest interpolation errors; among traditional methods, PCA is optimal

4. **Q4: Are these conclusions consistent across demo families?**
   - **Answer**: Representation performance rankings are completely consistent across 4 PDE types, demonstrating framework generalization

### Complex Field Specific Questions (Q5-Q10)
5. **Q5: For complex-valued PDEs (Helmholtz equation), which representation better captures frequency trajectories?**
   - **Answer**: **Real-imaginary representation is best for learning-based methods** (smoothness 7.914 vs 41.39, 5.2× advantage)

6. **Q6: How does phase unwrapping affect frequency trajectory smoothness and interpolability?**
   - **Answer**: Critical for traditional representations; still helpful but less critical for learning-based representations

7. **Q7: Does amplitude-phase representation provide better physical interpretability compared to real-imaginary?**
   - **Answer**: Yes, but representation quality is poorer, not suitable for learning-based methods

8. **Q8: For complex fields, are there intrinsic differences in how amplitude and phase vary with frequency compared to real and imaginary parts?**
   - **Answer**: Amplitude and phase vary more complexly (nonlinearly) and are harder to learn

9. **Q9: Can learning-based representations outperform traditional representations for complex field trajectories?**
   - **Answer**: **Yes, significantly outperforms** all traditional representations, validating neural network effectiveness

10. **Q10: What are the trade-offs between interpretability and performance?**
    - **Answer**: Clear trade-off: amplitude-phase has good interpretability but poor performance; learning-based has best performance but poor interpretability

## Experimental Findings

### 1. Traditional Representation Performance Ranking (Validated on 4 PDEs)
**PCA > Raw > Amplitude-phase > Fourier**
- Ranking consistent across phase_family, screened_poisson, helmholtz, and wave_equation
- PCA as theoretical limit of linear methods provides optimal performance
- Fourier representation least suitable for frequency trajectory modeling

### 2. Key Findings for Complex Helmholtz Equation
- **Learning-based representation advantage**: Autoencoder smoothness 10.04 vs PCA 66.63 (6.6× advantage)
- **Phase wrapping problem severe**: Unwrapped amplitude-phase representation intrinsic rank 27.79
- **Phase unwrapping effect limited**: Significant improvement but insufficient to make amplitude-phase better than real-imaginary

### 3. Autoencoder Design Comparison Core Conclusions
- **Best input design**: **Real-imaginary dual-channel images** (smoothness 7.914)
- **Second best**: Amplitude-phase input (with unwrapping, smoothness 41.39)
- **Worst**: Complex residual architecture (smoothness 159.5)
- **Architecture insight**: Simplicity outperforms complexity; 2-layer convolution sufficient

### 4. Value of Learning-based Methods
1. **Significantly reduces phase wrapping**: Traditional unwrapped smoothness 997.6 → Learning-based unwrapped 72.35 (13.8× improvement)
2. **Surpasses linear methods**: Autoencoders significantly outperform PCA and other linear dimensionality reduction methods
3. **Data efficient**: Only 12 training trajectories needed to learn effective representations

## Architecture

### Demo Families (PDE Solution Families)
1. **phase_family**: u_w(x,y) = A(x,y) * cos(w * τ(x,y) + φ(x,y))
2. **screened_poisson**: (Δ - α²) u_α = s(x,y) solved via FFT on periodic 2D domain
3. **helmholtz**: (∇² + k²) u = f(x,y) with complex field solutions, low frequency range (0.5-5.0) for meaningful trajectory variation
4. **wave_equation**: ∂²u/∂t² = c²∇²u with varying wave speed c

### Representation Modules
1. **Raw**: Field flattened as baseline (supports complex fields via real/imag concatenation)
2. **Fourier**: FFT coefficients (supports complex input)
3. **Amplitude-phase**: Hilbert transform extracts amplitude and phase (supports phase unwrapping for complex fields)
4. **PCA**: Linear dimensionality reduction (95% variance, supports complex fields via real/imag concatenation)
5. **Real-imag**: Complex field decomposed into real and imaginary parts
6. **Autoencoder**: Convolutional autoencoder learns nonlinear representation (real-imag input)
7. **Amplitude-phase autoencoder**: Autoencoder with amplitude and phase as input channels (supports phase unwrapping)
8. **Enhanced real-imag autoencoder**: Deeper autoencoder with residual connections (real-imag input)

### Metrics
1. **local_smoothness**: Local smoothness, evaluates smoothness of representation vectors with frequency changes (lower better)
2. **curvature/mean_curvature**: Curvature, evaluates nonlinearity (lower better)
3. **intrinsic_rank**: Intrinsic rank (participation ratio), evaluates effective dimensionality (lower better)
4. **interpolation_error**: Interpolation error, evaluates linear interpolation accuracy (lower better)

### Configuration System
Experiments are defined by YAML configs:
- `configs/demos/`: Demo family configurations
- `configs/experiments/`: Experiment configurations
- `configs/metrics/`: Metric configurations
- `configs/reprs/`: Representation configurations

### Output Structure
Results are saved under `outputs/` with subdirectories:
- `datasets/`       # Raw trajectory data
- `representations/` # Cached representation transforms
- `metrics/`        # Metric tables (per trajectory, aggregated)
- `figures/`        # Visualizations
- `tables/`         # Formatted result tables
- `summaries/`      # Markdown summaries

## Quick Start

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run a small experiment
```bash
python -m src.main --config configs/experiments/full_grid_small.yaml
```

### Run complex field autoencoder comparison
```bash
python -m src.main --config configs/experiments/autoencoder_comparison.yaml
```

### Generate trajectory GIFs for complex fields
```bash
python -m src.report.complex_trajectory_gif \
  --results outputs/experiment_name \
  --demo demo_name \
  --output outputs/gifs \
  --fps 3 \
  --multipanel
```

## Project Structure
```
freq-repr-study/
├── configs/           # Configuration files
├── src/               # Source code
│   ├── data_gen/      # PDE trajectory generation
│   ├── representations/ # Representation modules
│   ├── metrics/       # Metric computation
│   ├── experiments/   # Experiment orchestration
│   ├── report/        # Report and visualization
│   └── utils/         # Shared utilities
├── outputs/           # Experimental results (gitignored)
├── reports/           # Analysis reports
├── context/           # Research context and guidelines
├── agent/             # Agent workflow templates
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Detailed Reports
- **Final Summary**: `reports/final_summary_2026-03-19.md` - Complete research findings
- **Autoencoder Design Analysis**: `reports/autoencoder_design_analysis.md` - Detailed autoencoder comparison
- **Project Documentation**: `PROJECT_DOCUMENTATION.md` - Implementation details

## Citation
If you use this code in your research, please cite the project and include a link to the repository.

## License
[Specify your license here]

## Contact
[Your contact information]