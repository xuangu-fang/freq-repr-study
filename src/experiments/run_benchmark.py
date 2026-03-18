"""
Main benchmark orchestration.
"""

import os
import yaml
import numpy as np
import pickle
from tqdm import tqdm

from src.data_gen import (
    generate_phase_family_trajectory,
    generate_screened_poisson_trajectory,
)
from src.representations import (
    RawRepresentation,
    FourierRepresentation,
    AmplitudePhaseRepresentation,
    PCARepresentation,
)
from src.metrics import local_smoothness, mean_curvature, intrinsic_rank, interpolation_error


def run_benchmark(config):
    """
    Run the full benchmark according to the config.

    Parameters
    ----------
    config : dict
        Benchmark configuration dictionary.
    """
    # Set random seed
    seed = config.get("seed", 42)
    np.random.seed(seed)

    # Create output directory
    output_dir = config.get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Save config snapshot
    config_snapshot_path = os.path.join(output_dir, "config_snapshot.yaml")
    with open(config_snapshot_path, "w") as f:
        yaml.dump(config, f)

    # Load demo configs
    demo_configs = []
    for demo_path in config["demos"]:
        with open(demo_path, "r") as f:
            demo_cfg = yaml.safe_load(f)
        demo_configs.append(demo_cfg)

    # Initialize representations
    repr_names = config["representations"]
    representations = _init_representations(repr_names, config)

    # Initialize metrics
    metric_names = config["metrics"]
    metrics = _init_metrics(metric_names, config)

    # Dataset split
    n_traj = config["dataset"]["num_trajectories"]
    train_ratio = config["dataset"]["train_ratio"]
    val_ratio = config["dataset"]["val_ratio"]
    test_ratio = config["dataset"]["test_ratio"]

    n_train = int(n_traj * train_ratio)
    n_val = int(n_traj * val_ratio)
    n_test = n_traj - n_train - n_val

    split_indices = {
        "train": list(range(n_train)),
        "val": list(range(n_train, n_train + n_val)),
        "test": list(range(n_train + n_val, n_traj)),
    }

    # Results storage
    results = {
        "config": config,
        "seed": seed,
        "demo_results": {},
    }

    # For each demo family
    for demo_cfg in demo_configs:
        demo_name = demo_cfg["name"]
        print(f"Processing demo family: {demo_name}")

        demo_results = {
            "trajectories": {},
            "representations": {},
            "metrics": {},
        }

        # Generate trajectories
        trajectories = []
        frequencies_list = []
        metadata_list = []

        for i in tqdm(range(n_traj), desc=f"Generating {demo_name} trajectories"):
            traj, freqs, meta = _generate_trajectory(demo_cfg)
            trajectories.append(traj)
            frequencies_list.append(freqs)
            metadata_list.append(meta)

        demo_results["trajectories"]["raw"] = trajectories
        demo_results["trajectories"]["frequencies"] = frequencies_list
        demo_results["trajectories"]["metadata"] = metadata_list

        # Split trajectories
        for split_name, indices in split_indices.items():
            demo_results["trajectories"][split_name] = [
                trajectories[i] for i in indices
            ]
            demo_results["trajectories"][f"{split_name}_frequencies"] = [
                frequencies_list[i] for i in indices
            ]

        # For each representation
        for repr_name, repr_obj in representations.items():
            print(f"  Applying representation: {repr_name}")

            # Fit on training trajectories
            train_trajs = demo_results["trajectories"]["train"]
            # Flatten all fields from all training trajectories
            flat_train_fields = [field for traj in train_trajs for field in traj]
            repr_obj.fit(flat_train_fields)

            # Transform all trajectories
            repr_trajectories = []
            for traj in trajectories:
                repr_traj = repr_obj.transform(traj)
                repr_trajectories.append(repr_traj)

            demo_results["representations"][repr_name] = {
                "object": repr_obj,
                "transformed": repr_trajectories,
                "metadata": repr_obj.metadata(),
            }

            # Split transformed trajectories
            for split_name, indices in split_indices.items():
                key = f"{repr_name}_{split_name}"
                demo_results["representations"][key] = [
                    repr_trajectories[i] for i in indices
                ]

        # Compute metrics for each representation
        for repr_name in representations.keys():
            repr_trajectories = demo_results["representations"][repr_name]["transformed"]
            metric_results = {}

            for metric_name, metric_func in metrics.items():
                # Compute metric per trajectory
                per_traj_values = []
                for traj, freqs in zip(repr_trajectories, frequencies_list):
                    value = metric_func(traj, freqs)
                    per_traj_values.append(value)

                metric_results[metric_name] = per_traj_values

                # Aggregate per split
                for split_name, indices in split_indices.items():
                    split_values = [per_traj_values[i] for i in indices]
                    metric_results[f"{metric_name}_{split_name}"] = split_values

            demo_results["metrics"][repr_name] = metric_results

        results["demo_results"][demo_name] = demo_results

        # Save per‑demo results
        demo_output_dir = os.path.join(output_dir, "demo_results", demo_name)
        os.makedirs(demo_output_dir, exist_ok=True)
        with open(os.path.join(demo_output_dir, "results.pkl"), "wb") as f:
            pickle.dump(demo_results, f)

    # Save global results
    results_path = os.path.join(output_dir, "benchmark_results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    # Generate summary
    _generate_summary(results, output_dir)

    print(f"Benchmark completed. Results saved to {output_dir}")


def _init_representations(repr_names, config):
    """Initialize representation objects based on names."""
    repr_map = {
        "raw": RawRepresentation,
        "fourier": FourierRepresentation,
        "amplitude_phase": AmplitudePhaseRepresentation,
        "pca": PCARepresentation,
        # Add other representations here
    }

    representations = {}
    for name in repr_names:
        if name not in repr_map:
            raise ValueError(f"Unknown representation: {name}")
        representations[name] = repr_map[name]()
    return representations


def _init_metrics(metric_names, config):
    """Initialize metric functions based on names."""
    metric_map = {
        "local_smoothness": local_smoothness,
        "curvature": mean_curvature,
        "intrinsic_rank": intrinsic_rank,
        "interpolation_error": interpolation_error,
        # Add other metrics here
    }

    metrics = {}
    for name in metric_names:
        if name not in metric_map:
            raise ValueError(f"Unknown metric: {name}")
        metrics[name] = metric_map[name]
    return metrics


def _generate_trajectory(demo_config):
    """Generate a single trajectory based on demo config."""
    demo_type = demo_config.get("name", "")
    if demo_type == "phase_family":
        return generate_phase_family_trajectory(demo_config)
    elif demo_type == "screened_poisson":
        return generate_screened_poisson_trajectory(demo_config)
    else:
        raise ValueError(f"Unknown demo type: {demo_type}")


def _generate_summary(results, output_dir):
    """Generate a text summary of results."""
    summary_path = os.path.join(output_dir, "summary.md")
    with open(summary_path, "w") as f:
        f.write("# Benchmark Summary\n\n")
        f.write(f"Seed: {results['seed']}\n\n")

        for demo_name, demo_res in results["demo_results"].items():
            f.write(f"## Demo: {demo_name}\n\n")

            for repr_name in demo_res["representations"].keys():
                if repr_name.endswith("_train") or repr_name.endswith("_val") or repr_name.endswith("_test") or repr_name in ["object", "metadata", "transformed"]:
                    continue
                f.write(f"### Representation: {repr_name}\n")

                for metric_name, metric_values in demo_res["metrics"][repr_name].items():
                    if metric_name.endswith("_train") or metric_name.endswith("_val") or metric_name.endswith("_test"):
                        continue  # Skip split-specific entries
                    values = metric_values
                    if isinstance(values, list) and values:
                        if isinstance(values[0], (np.ndarray, list)):
                            # Flatten if needed
                            flat = np.concatenate([np.array(v).flatten() for v in values])
                            mean_val = np.mean(flat)
                            std_val = np.std(flat)
                        else:
                            mean_val = np.mean(values)
                            std_val = np.std(values)
                        f.write(f"- {metric_name}: {mean_val:.4g} ± {std_val:.4g}\n")
                    else:
                        f.write(f"- {metric_name}: {values}\n")
                f.write("\n")

        f.write("\n---\n")
        f.write("Generated by benchmark script.\n")