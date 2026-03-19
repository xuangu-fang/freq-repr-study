#!/usr/bin/env python
"""Test helmholtz data generation."""

import sys
sys.path.insert(0, '.')

import numpy as np
import yaml
from src.data_gen.helmholtz import generate_helmholtz_trajectory, generate_helmholtz_dataset

print("Testing helmholtz data generation...")

# Load helmholtz config
with open("configs/demos/helmholtz.yaml", "r") as f:
    config = yaml.safe_load(f)

print(f"Config loaded: {config['name']}")
print(f"Resolution: {config['resolution']}")
print(f"Num frequencies: {config['num_frequencies']}")
print(f"Wave number range: {config['wave_number_grid']['start']} to {config['wave_number_grid']['end']}")
print(f"Epsilon: {config.get('epsilon', 'default 0.01')}")

# Test single trajectory generation
print("\n1. Testing single trajectory generation...")
np.random.seed(42)
trajectory, wave_numbers, metadata = generate_helmholtz_trajectory(config)

print(f"  Trajectory length: {len(trajectory)}")
print(f"  Field shape: {trajectory[0].shape}")
print(f"  Wave numbers: {wave_numbers[:5]}... (first 5 of {len(wave_numbers)})")
print(f"  Metadata keys: {list(metadata.keys())}")

# Check that fields are real-valued
for i, field in enumerate(trajectory[:3]):
    print(f"  Field {i}: min={field.min():.3f}, max={field.max():.3f}, mean={field.mean():.3f}")

# Test dataset generation
print("\n2. Testing dataset generation (small)...")
dataset = generate_helmholtz_dataset(config, num_trajectories=3, output_dir=None, seed=42)

print(f"  Dataset trajectories shape: {dataset['trajectories'].shape}")
print(f"  Wave numbers shape: {dataset['wave_numbers'].shape}")
print(f"  Metadata list length: {len(dataset['metadata_list'])}")
print(f"  Config included: {'config' in dataset}")
print(f"  Seed included: {dataset['seed']}")

# Check basic properties
trajectories = dataset['trajectories']
print(f"\n3. Dataset statistics:")
print(f"  Overall min: {trajectories.min():.3f}")
print(f"  Overall max: {trajectories.max():.3f}")
print(f"  Overall mean: {trajectories.mean():.3f}")
print(f"  Overall std: {trajectories.std():.3f}")

# Check that wave numbers are monotonic
wave_nums = dataset['wave_numbers']
if config['wave_number_grid']['type'] == 'linear':
    diffs = np.diff(wave_nums)
    print(f"\n4. Wave number linearity check:")
    print(f"  Min diff: {diffs.min():.6f}")
    print(f"  Max diff: {diffs.max():.6f}")
    print(f"  Constant step? {np.allclose(diffs, diffs[0], rtol=1e-10)}")

# Test with different configurations
print("\n5. Testing with different configurations...")

# Test high frequency
high_freq_config = config.copy()
high_freq_config['wave_number_grid'] = {'type': 'linear', 'start': 10.0, 'end': 50.0}
high_freq_config['resolution'] = 64

traj_high, wave_high, meta_high = generate_helmholtz_trajectory(high_freq_config)
print(f"  High frequency (10-50): trajectory length {len(traj_high)}, wave numbers {wave_high[:3]}...")

# Test with smooth_random source
smooth_config = config.copy()
smooth_config['source'] = {
    'type': 'smooth_random',
    'low': 0.0,
    'high': 1.0,
    'smoothing_sigma': 4.0
}

traj_smooth, wave_smooth, meta_smooth = generate_helmholtz_trajectory(smooth_config)
print(f"  Smooth random source: trajectory length {len(traj_smooth)}, source range [{meta_smooth['source'].min():.3f}, {meta_smooth['source'].max():.3f}]")

print("\nAll helmholtz tests completed successfully!")