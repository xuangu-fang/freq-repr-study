"""
Wave equation demo: ∂²u/∂t² = c²∇²u + f(x,y) on a periodic 2D box.
Implemented in frequency domain, reduces to Helmholtz equation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from .helmholtz import generate_helmholtz_trajectory as _generate_helmholtz_trajectory
from .helmholtz import generate_helmholtz_dataset as _generate_helmholtz_dataset


def generate_wave_equation_trajectory(config):
    """
    Generate a trajectory of fields for the wave equation in frequency domain.

    Parameters
    ----------
    config : dict
        Configuration dictionary with keys:
        - resolution: int (grid size)
        - num_frequencies: int
        - frequency_grid: dict with type, start, end (frequency ω values)
        - wave_speed: float or dict (optional, default=1.0)
        - source: dict with type, low, high, smoothing_sigma
        - epsilon: float (optional, default=0.01) small imaginary part to avoid resonance

    Returns
    -------
    trajectory : list of ndarray
        List of 2D fields (resolution x resolution) for each ω
    frequencies : ndarray
        Array of frequency ω values
    metadata : dict
        Metadata including the source field, wave numbers, and wave speed
    """
    # Extract wave speed configuration
    wave_speed_cfg = config.get("wave_speed", 1.0)
    if isinstance(wave_speed_cfg, dict):
        # If wave_speed is a dict, it may have type: 'constant' or 'random'
        wave_speed_type = wave_speed_cfg.get("type", "constant")
        if wave_speed_type == "constant":
            c = wave_speed_cfg.get("value", 1.0)
        elif wave_speed_type == "random":
            # Random wave speed for each frequency? Or fixed per trajectory?
            # For simplicity, fixed per trajectory
            c = np.random.uniform(wave_speed_cfg.get("min", 0.5),
                                 wave_speed_cfg.get("max", 2.0))
        else:
            raise ValueError(f"Unsupported wave_speed type: {wave_speed_type}")
    else:
        # Constant wave speed
        c = float(wave_speed_cfg)

    # Create a Helmholtz-compatible config
    helmholtz_config = config.copy()

    # Convert frequency grid to wave number grid: k = ω/c
    if "frequency_grid" in helmholtz_config:
        freq_cfg = helmholtz_config["frequency_grid"]
        wave_number_cfg = freq_cfg.copy()
        # The start and end values are frequencies ω, convert to k = ω/c
        wave_number_cfg["start"] = freq_cfg["start"] / c
        wave_number_cfg["end"] = freq_cfg["end"] / c
        helmholtz_config["wave_number_grid"] = wave_number_cfg
        # Remove frequency_grid to avoid confusion
        if "frequency_grid" in helmholtz_config:
            del helmholtz_config["frequency_grid"]
    elif "wave_number_grid" in helmholtz_config:
        # If already wave_number_grid, use as is (assuming k = ω/c already)
        pass
    else:
        raise ValueError("Config must contain either 'frequency_grid' or 'wave_number_grid'")

    # Generate trajectory using Helmholtz solver
    trajectory, wave_numbers, helmholtz_meta = _generate_helmholtz_trajectory(helmholtz_config)

    # Convert wave numbers back to frequencies: ω = k*c
    frequencies = wave_numbers * c

    # Add wave speed to metadata
    metadata = helmholtz_meta.copy()
    metadata["wave_speed"] = c
    metadata["frequencies"] = frequencies
    metadata["wave_numbers"] = wave_numbers

    return trajectory, frequencies, metadata


def generate_wave_equation_dataset(config, num_trajectories, output_dir=None, seed=42):
    """
    Generate a dataset of wave equation trajectories.

    Parameters
    ----------
    config : dict
        Configuration dictionary for wave equation (same as generate_wave_equation_trajectory)
    num_trajectories : int
        Number of trajectories to generate
    output_dir : str, optional
        Directory to save dataset and previews. If None, data is returned but not saved.
    seed : int
        Random seed for reproducibility

    Returns
    -------
    dataset : dict
        Dictionary containing:
        - trajectories: ndarray of shape (num_trajectories, num_frequencies, resolution, resolution)
        - frequencies: ndarray of shape (num_frequencies,)
        - metadata_list: list of metadata dicts for each trajectory
        - config: input configuration
        - seed: random seed used
    """
    np.random.seed(seed)

    r = config["resolution"]
    n_freq = config["num_frequencies"]

    # Create Helmholtz-compatible config for generating consistent wave numbers
    helmholtz_config = config.copy()
    wave_speed_cfg = config.get("wave_speed", 1.0)

    # Handle wave speed (constant or random)
    if isinstance(wave_speed_cfg, dict):
        wave_speed_type = wave_speed_cfg.get("type", "constant")
    else:
        wave_speed_type = "constant"

    # Initialize arrays
    trajectories = np.zeros((num_trajectories, n_freq, r, r), dtype=np.float32)
    metadata_list = []
    wave_speeds = []

    # For constant wave speed, compute frequencies once
    if wave_speed_type == "constant":
        if isinstance(wave_speed_cfg, dict):
            c = wave_speed_cfg.get("value", 1.0)
        else:
            c = float(wave_speed_cfg)

        # Create frequency grid
        if "frequency_grid" in config:
            freq_cfg = config["frequency_grid"]
            if freq_cfg["type"] == "linear":
                frequencies = np.linspace(freq_cfg["start"], freq_cfg["end"], n_freq)
            elif freq_cfg["type"] == "log":
                frequencies = np.logspace(np.log10(freq_cfg["start"]), np.log10(freq_cfg["end"]), n_freq)
            else:
                raise ValueError(f"Unsupported grid type: {freq_cfg['type']}")
        else:
            raise ValueError("Config must contain 'frequency_grid' for wave equation")

        # Generate each trajectory with same wave speed
        for i in range(num_trajectories):
            traj, freqs, meta = generate_wave_equation_trajectory(config)
            trajectories[i] = np.array(traj)
            metadata_list.append(meta)
            wave_speeds.append(c)

    else:  # random wave speed per trajectory
        # Generate each trajectory with its own random wave speed
        for i in range(num_trajectories):
            # Create trajectory-specific config with random wave speed
            traj_config = config.copy()
            if isinstance(wave_speed_cfg, dict):
                c = np.random.uniform(wave_speed_cfg.get("min", 0.5),
                                     wave_speed_cfg.get("max", 2.0))
                traj_config["wave_speed"] = c
            else:
                c = float(wave_speed_cfg)

            traj, freqs, meta = generate_wave_equation_trajectory(traj_config)

            # For first trajectory, set frequencies array
            if i == 0:
                frequencies = freqs
                trajectories[i] = np.array(traj)
            else:
                # Check frequencies match (should for same config except wave speed)
                if not np.allclose(freqs, frequencies, rtol=1e-10):
                    # If wave speed varies, frequencies will differ
                    # We need to interpolate to common frequency grid
                    # For simplicity, we'll store as-is and handle in analysis
                    pass
                trajectories[i] = np.array(traj)

            metadata_list.append(meta)
            wave_speeds.append(c)

    # Create dataset dictionary
    dataset = {
        "trajectories": trajectories,      # shape (N, M, H, W)
        "frequencies": frequencies,        # shape (M,)
        "wave_speeds": np.array(wave_speeds),  # shape (N,)
        "metadata_list": metadata_list,    # list of N dicts
        "config": config,
        "seed": seed,
        "num_trajectories": num_trajectories,
    }

    # Save dataset if output_dir is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        # Save dataset as numpy file
        dataset_path = os.path.join(output_dir, "wave_equation_dataset.npz")
        np.savez_compressed(
            dataset_path,
            trajectories=trajectories,
            frequencies=frequencies,
            wave_speeds=np.array(wave_speeds),
            config=config,
            seed=seed,
            num_trajectories=num_trajectories,
        )

        # Save metadata separately (as it contains arrays)
        metadata_path = os.path.join(output_dir, "wave_equation_metadata.pkl")
        import pickle
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata_list, f)

        # Generate preview plots
        preview_dir = os.path.join(output_dir, "preview")
        os.makedirs(preview_dir, exist_ok=True)
        _generate_preview_plots(dataset, preview_dir, num_previews=5)

        print(f"Dataset saved to {output_dir}")
        print(f"  - Trajectories shape: {trajectories.shape}")
        print(f"  - Frequencies: {frequencies}")
        print(f"  - Wave speeds: {np.array(wave_speeds)}")
        print(f"  - Config saved with dataset")
        print(f"  - Preview plots saved to {preview_dir}")

    return dataset


def _generate_preview_plots(dataset, output_dir, num_previews=5):
    """
    Generate preview plots for the wave equation dataset.

    Parameters
    ----------
    dataset : dict
        Dataset dictionary returned by generate_wave_equation_dataset
    output_dir : str
        Directory to save preview plots
    num_previews : int
        Number of trajectories to preview
    """
    trajectories = dataset["trajectories"]
    frequencies = dataset["frequencies"]
    wave_speeds = dataset["wave_speeds"]
    num_trajectories = dataset["num_trajectories"]

    # Limit number of previews
    num_previews = min(num_previews, num_trajectories)

    # Create previews for first num_previews trajectories
    for i in range(num_previews):
        traj = trajectories[i]
        c = wave_speeds[i] if i < len(wave_speeds) else 1.0

        # Create figure with 2x3 subplots (show 6 frequency points)
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()

        # Show 6 frequency points (or all if less than 6)
        n_show = min(6, len(frequencies))
        step = max(1, len(frequencies) // n_show)

        for j, ax_idx in enumerate(range(0, len(frequencies), step)):
            if j >= n_show:
                break

            freq_idx = ax_idx
            field = traj[freq_idx]
            freq_val = frequencies[freq_idx]

            im = axes[j].imshow(field, cmap='RdBu_r', origin='lower')
            axes[j].set_title(f"ω = {freq_val:.2f}, c={c:.2f}")
            axes[j].axis('off')
            plt.colorbar(im, ax=axes[j], fraction=0.046, pad=0.04)

        # Hide unused subplots
        for j in range(n_show, 6):
            axes[j].axis('off')

        fig.suptitle(f"Wave Equation Trajectory {i+1}/{num_trajectories}", fontsize=14)
        plt.tight_layout()

        # Save figure
        fig_path = os.path.join(output_dir, f"trajectory_{i+1:03d}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Create summary plot showing first field of each preview trajectory
    fig, axes = plt.subplots(1, num_previews, figsize=(4*num_previews, 4))
    if num_previews == 1:
        axes = [axes]

    for i in range(num_previews):
        field = trajectories[i, 0]  # First frequency
        c = wave_speeds[i] if i < len(wave_speeds) else 1.0
        im = axes[i].imshow(field, cmap='RdBu_r', origin='lower')
        axes[i].set_title(f"Traj {i+1}, ω={frequencies[0]:.2f}, c={c:.2f}")
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    fig.suptitle(f"First Frequency (ω={frequencies[0]:.2f}) for {num_previews} Trajectories", fontsize=14)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "summary_first_frequency.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  Generated {num_previews + 1} preview plots in {output_dir}")