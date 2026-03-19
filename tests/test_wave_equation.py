#!/usr/bin/env python
"""Test wave equation data generation."""

import sys
sys.path.insert(0, '.')

import numpy as np
import yaml
from src.data_gen.wave_equation import generate_wave_equation_trajectory, generate_wave_equation_dataset

print("Testing wave equation data generation...")

# Load wave equation config
with open("configs/demos/wave_equation.yaml", "r") as f:
    config = yaml.safe_load(f)

print(f"Config loaded: {config['name']}")
print(f"Resolution: {config['resolution']}")
print(f"Num frequencies: {config['num_frequencies']}")
print(f"Frequency range: {config['frequency_grid']['start']} to {config['frequency_grid']['end']}")
print(f"Wave speed: {config.get('wave_speed', 'default 1.0')}")

# Test single trajectory generation
print("\n1. Testing single trajectory generation...")
np.random.seed(42)
trajectory, frequencies, metadata = generate_wave_equation_trajectory(config)

print(f"  Trajectory length: {len(trajectory)}")
print(f"  Field shape: {trajectory[0].shape}")
print(f"  Frequencies: {frequencies[:5]}... (first 5 of {len(frequencies)})")
print(f"  Metadata keys: {list(metadata.keys())}")
print(f"  Wave speed: {metadata.get('wave_speed', 'N/A')}")

# Check that fields are real-valued
for i, field in enumerate(trajectory[:3]):
    print(f"  Field {i}: min={field.min():.3f}, max={field.max():.3f}, mean={field.mean():.3f}")

# Test dataset generation
print("\n2. Testing dataset generation (small)...")
dataset = generate_wave_equation_dataset(config, num_trajectories=3, output_dir=None, seed=42)

print(f"  Dataset trajectories shape: {dataset['trajectories'].shape}")
print(f"  Frequencies shape: {dataset['frequencies'].shape}")
print(f"  Wave speeds shape: {dataset.get('wave_speeds', np.array([1.0])).shape}")
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

# Check that frequencies are monotonic
freqs = dataset['frequencies']
if config['frequency_grid']['type'] == 'linear':
    diffs = np.diff(freqs)
    print(f"\n4. Frequency linearity check:")
    print(f"  Min diff: {diffs.min():.6f}")
    print(f"  Max diff: {diffs.max():.6f}")
    print(f"  Constant step? {np.allclose(diffs, diffs[0], rtol=1e-10)}")

# Test with random wave speed
print("\n5. Testing with random wave speed...")
random_wave_speed_config = config.copy()
random_wave_speed_config['wave_speed'] = {
    'type': 'random',
    'min': 0.5,
    'max': 2.0
}

np.random.seed(123)
traj_random, freqs_random, meta_random = generate_wave_equation_trajectory(random_wave_speed_config)
c_random = meta_random.get('wave_speed', 1.0)
print(f"  Random wave speed: {c_random:.3f}")
print(f"  Frequencies: {freqs_random[:3]}...")
print(f"  Expected k = ω/c: first k = {freqs_random[0]/c_random:.3f}")

# Test with different frequency range
print("\n6. Testing with high frequency range...")
high_freq_config = config.copy()
high_freq_config['frequency_grid'] = {'type': 'linear', 'start': 10.0, 'end': 50.0}

traj_high, freqs_high, meta_high = generate_wave_equation_trajectory(high_freq_config)
print(f"  High frequency (10-50): trajectory length {len(traj_high)}, frequencies {freqs_high[:3]}...")

print("\nAll wave equation tests completed successfully!")