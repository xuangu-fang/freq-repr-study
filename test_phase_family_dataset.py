#!/usr/bin/env python
"""Test phase_family dataset generation."""

import sys
sys.path.insert(0, '.')

import yaml
import numpy as np
from src.data_gen import generate_phase_family_dataset

# Load config
config_path = "configs/demos/phase_family.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

print("Testing phase_family dataset generation...")
print(f"Config: {config['name']}")
print(f"Resolution: {config['resolution']}")
print(f"Num frequencies: {config['num_frequencies']}")

# Generate small dataset
output_dir = "outputs/datasets/phase_family_preview"
dataset = generate_phase_family_dataset(
    config=config,
    num_trajectories=10,  # 10 trajectories for testing
    output_dir=output_dir,
    seed=42
)

# Verify dataset structure
print("\nDataset verification:")
print(f"Trajectories shape: {dataset['trajectories'].shape}")
print(f"Expected: (10, {config['num_frequencies']}, {config['resolution']}, {config['resolution']})")
print(f"Frequencies shape: {dataset['frequencies'].shape}")
print(f"Frequencies: {dataset['frequencies']}")
print(f"Number of metadata entries: {len(dataset['metadata_list'])}")

# Check that shapes match
assert dataset['trajectories'].shape == (10, config['num_frequencies'], config['resolution'], config['resolution'])
assert dataset['frequencies'].shape == (config['num_frequencies'],)
assert len(dataset['metadata_list']) == 10

print("\nAll checks passed!")
print(f"Dataset saved to: {output_dir}")
print(f"Preview plots saved to: {output_dir}/preview")

# Quick statistical check
traj_mean = np.mean(dataset['trajectories'])
traj_std = np.std(dataset['trajectories'])
print(f"\nDataset statistics:")
print(f"  Mean: {traj_mean:.4f}")
print(f"  Std: {traj_std:.4f}")
print(f"  Min: {np.min(dataset['trajectories']):.4f}")
print(f"  Max: {np.max(dataset['trajectories']):.4f}")