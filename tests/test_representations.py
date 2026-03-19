#!/usr/bin/env python
"""Test representation modules."""

import sys
sys.path.insert(0, '.')

import numpy as np
from src.representations import raw_repr, fourier_repr, amplitude_phase_repr, pca_repr

print("Testing representation modules...")

# Create a sample trajectory: 5 frequencies, 8x8 fields
np.random.seed(42)
sample_array = np.random.randn(5, 8, 8)
sample_traj = list(sample_array)  # Convert to list of 2D fields

print(f"Sample trajectory: {len(sample_traj)} fields, each shape {sample_traj[0].shape}")

# Test raw representation
print("\n1. Testing raw representation:")
raw_module = raw_repr.RawRepresentation()
print(f"  Metadata: {raw_module.metadata()}")

# Transform
raw_traj = raw_module.transform(sample_traj)
print(f"  Transformed: {len(raw_traj)} vectors, each shape {raw_traj[0].shape}")

# Inverse transform (if available)
if hasattr(raw_module, 'inverse_transform'):
    inv_traj = raw_module.inverse_transform(raw_traj)
    print(f"  Inverse transformed: {len(inv_traj)} fields, each shape {inv_traj[0].shape}")
    # Check reconstruction error
    recon_error = np.mean([np.mean(np.abs(inv_traj[i] - sample_traj[i])) for i in range(len(sample_traj))])
    print(f"  Reconstruction error (MAE): {recon_error:.6f}")

# Test fourier representation
print("\n2. Testing fourier representation:")
fourier_module = fourier_repr.FourierRepresentation()
print(f"  Metadata: {fourier_module.metadata()}")

# Transform
fourier_traj = fourier_module.transform(sample_traj)
print(f"  Transformed: {len(fourier_traj)} vectors, each shape {fourier_traj[0].shape}")

# Inverse transform
if hasattr(fourier_module, 'inverse_transform'):
    inv_traj = fourier_module.inverse_transform(fourier_traj)
    print(f"  Inverse transformed: {len(inv_traj)} fields, each shape {inv_traj[0].shape}")
    # Check reconstruction error
    recon_error = np.mean([np.mean(np.abs(inv_traj[i] - sample_traj[i])) for i in range(len(sample_traj))])
    print(f"  Reconstruction error (MAE): {recon_error:.6f}")

# Test amplitude-phase representation
print("\n3. Testing amplitude-phase representation:")
amp_phase_module = amplitude_phase_repr.AmplitudePhaseRepresentation()
print(f"  Metadata: {amp_phase_module.metadata()}")

# Transform
amp_phase_traj = amp_phase_module.transform(sample_traj)
print(f"  Transformed: {len(amp_phase_traj)} vectors, each shape {amp_phase_traj[0].shape}")

# Inverse transform
if hasattr(amp_phase_module, 'inverse_transform'):
    inv_traj = amp_phase_module.inverse_transform(amp_phase_traj)
    print(f"  Inverse transformed: {len(inv_traj)} fields, each shape {inv_traj[0].shape}")
    # Check reconstruction error
    recon_error = np.mean([np.mean(np.abs(inv_traj[i] - sample_traj[i])) for i in range(len(sample_traj))])
    print(f"  Reconstruction error (MAE): {recon_error:.6f}")

# Test PCA representation
print("\n4. Testing PCA representation:")
pca_module = pca_repr.PCARepresentation({'n_components': 0.95})
print(f"  Metadata before fit: {pca_module.metadata()}")

# Fit on sample data (flatten all fields)
pca_module.fit(sample_traj)
print(f"  Metadata after fit: {pca_module.metadata()}")

# Transform
pca_traj = pca_module.transform(sample_traj)
print(f"  Transformed: {len(pca_traj)} vectors, each shape {pca_traj[0].shape}")

# Inverse transform
if hasattr(pca_module, 'inverse_transform'):
    inv_traj = pca_module.inverse_transform(pca_traj)
    print(f"  Inverse transformed: {len(inv_traj)} fields, each shape {inv_traj[0].shape}")
    # Check reconstruction error
    recon_error = np.mean([np.mean(np.abs(inv_traj[i] - sample_traj[i])) for i in range(len(sample_traj))])
    print(f"  Reconstruction error (MAE): {recon_error:.6f}")

print("\nAll representation tests completed!")