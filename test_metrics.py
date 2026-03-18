#!/usr/bin/env python
"""Test metric modules."""

import sys
sys.path.insert(0, '.')

import numpy as np
from src.metrics import local_smoothness, mean_curvature, intrinsic_rank, interpolation_error

print("Testing metric modules...")

# Create a sample trajectory: 5 frequencies, 10-dimensional representation vectors
np.random.seed(42)
sample_traj = [np.random.randn(10) for _ in range(5)]
frequencies = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

print(f"Sample trajectory: {len(sample_traj)} vectors, each dimension {sample_traj[0].shape[0]}")
print(f"Frequencies: {frequencies}")

# Test local smoothness
print("\n1. Testing local smoothness:")
smoothness = local_smoothness(sample_traj, frequencies)
print(f"  Local smoothness: {smoothness:.6f}")

# Test mean curvature
print("\n2. Testing mean curvature:")
curv = mean_curvature(sample_traj, frequencies)
print(f"  Mean curvature: {curv:.6f}")

# Test intrinsic rank
print("\n3. Testing intrinsic rank:")
rank_val = intrinsic_rank(sample_traj, frequencies)
print(f"  Intrinsic rank (participation ratio): {rank_val:.6f}")

# Test interpolation error
print("\n4. Testing interpolation error:")
interp_err = interpolation_error(sample_traj, frequencies)
print(f"  Interpolation error: {interp_err:.6f}")

# Additional test: linear trajectory should have low curvature and interpolation error
print("\n5. Testing with linear trajectory:")
# Create a linear trajectory: r(w) = a + b*w
a = np.random.randn(10)
b = np.random.randn(10)
linear_traj = [a + b * w for w in frequencies]
print(f"  Linear trajectory created")

lin_smoothness = local_smoothness(linear_traj, frequencies)
lin_curv = mean_curvature(linear_traj, frequencies)
lin_rank = intrinsic_rank(linear_traj, frequencies)
lin_interp = interpolation_error(linear_traj, frequencies)

print(f"  Linear trajectory metrics:")
print(f"    Smoothness: {lin_smoothness:.6f}")
print(f"    Mean curvature: {lin_curv:.6f} (should be near zero)")
print(f"    Intrinsic rank: {lin_rank:.6f} (should be near 1)")
print(f"    Interpolation error: {lin_interp:.6f} (should be near zero)")

print("\nAll metric tests completed!")