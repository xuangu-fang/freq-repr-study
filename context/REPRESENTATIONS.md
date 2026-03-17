# Representations

## raw
Flatten the original field.

## fourier
Use FFT coefficients or selected spectral modes.

## amplitude_phase
For oscillatory fields, represent the field by amplitude and phase.

## pca
Fit PCA on training trajectories and project fields into latent space.

## Rules
Each representation module must implement:
- fit(train_data, config) if needed
- transform(trajectory)
- inverse_transform(...) if available
- metadata()