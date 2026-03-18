#!/usr/bin/env python
"""Check screened_poisson config compatibility."""

import sys
sys.path.insert(0, '.')

import yaml

# Load config
config_path = "configs/demos/screened_poisson.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

print("Current config:")
print(yaml.dump(config, default_flow_style=False))

print("\nChecking compatibility with generate_screened_poisson_trajectory...")
print(f"Config keys: {list(config.keys())}")

# Check required keys
required_keys = ["resolution", "num_frequencies", "frequency_grid", "source"]
for key in required_keys:
    if key not in config:
        print(f"  Missing: {key}")
    else:
        print(f"  Found: {key}")

# Check frequency_grid vs alpha_grid
if "alpha_grid" in config and "frequency_grid" not in config:
    print("\nNote: Config uses 'alpha_grid' but code expects 'frequency_grid'")

# Check source type
if "source" in config:
    print(f"Source config: {config['source']}")