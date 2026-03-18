"""
Phase family demo: u_w(x,y) = A(x,y) * cos(w * tau(x,y) + phi(x,y))
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def generate_phase_family_trajectory(config):
    """
    Generate a trajectory of fields for the phase family.

    Parameters
    ----------
    config : dict
        Configuration dictionary with keys:
        - resolution: int (grid size)
        - num_frequencies: int
        - frequency_grid: dict with type, start, end
        - amplitude: dict with type, low, high, smoothing_sigma
        - tau: dict with type, low, high, smoothing_sigma
        - phi: dict with type, low, high, smoothing_sigma

    Returns
    -------
    trajectory : list of ndarray
        List of 2D fields (resolution x resolution) for each frequency
    frequencies : ndarray
        Array of frequency values
    metadata : dict
        Metadata including the generated A, tau, phi fields
    """
    r = config["resolution"]
    n_freq = config["num_frequencies"]

    # Create frequency grid
    freq_cfg = config["frequency_grid"]
    if freq_cfg["type"] == "linear":
        frequencies = np.linspace(freq_cfg["start"], freq_cfg["end"], n_freq)
    else:
        raise ValueError(f"Unsupported frequency grid type: {freq_cfg['type']}")

    # Generate smooth random fields
    A = _smooth_random_field(r, config["amplitude"])
    tau = _smooth_random_field(r, config["tau"])
    phi = _smooth_random_field(r, config["phi"])

    # Create coordinate grid
    x = np.linspace(0, 2 * np.pi, r, endpoint=False)
    y = np.linspace(0, 2 * np.pi, r, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    trajectory = []
    for w in frequencies:
        field = A * np.cos(w * tau + phi)
        trajectory.append(field)

    metadata = {
        "A": A,
        "tau": tau,
        "phi": phi,
        "X": X,
        "Y": Y,
    }
    return trajectory, frequencies, metadata


def _smooth_random_field(resolution, config):
    """
    Generate a smooth random field.

    Parameters
    ----------
    resolution : int
        Grid size
    config : dict
        Must contain keys: low, high, smoothing_sigma

    Returns
    -------
    field : ndarray
        2D array of shape (resolution, resolution)
    """
    low = config["low"]
    high = config["high"]
    sigma = config["smoothing_sigma"]

    # Uniform random noise
    noise = np.random.uniform(low, high, (resolution, resolution))

    # Apply Gaussian smoothing
    smoothed = gaussian_filter(noise, sigma=sigma, mode="wrap")

    return smoothed


def generate_phase_family_dataset(config, num_trajectories, output_dir=None, seed=42):
    """
    Generate a dataset of phase family trajectories.

    Parameters
    ----------
    config : dict
        Configuration dictionary for phase family (same as generate_phase_family_trajectory)
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

    # Create frequency grid (same for all trajectories)
    freq_cfg = config["frequency_grid"]
    if freq_cfg["type"] == "linear":
        frequencies = np.linspace(freq_cfg["start"], freq_cfg["end"], n_freq)
    else:
        raise ValueError(f"Unsupported frequency grid type: {freq_cfg['type']}")

    # Initialize arrays
    trajectories = np.zeros((num_trajectories, n_freq, r, r), dtype=np.float32)
    metadata_list = []

    # Generate each trajectory
    for i in range(num_trajectories):
        traj, freqs, meta = generate_phase_family_trajectory(config)
        # Convert list to array and store
        trajectories[i] = np.array(traj)
        metadata_list.append(meta)

    # Create dataset dictionary
    dataset = {
        "trajectories": trajectories,      # shape (N, M, H, W)
        "frequencies": frequencies,        # shape (M,)
        "metadata_list": metadata_list,    # list of N dicts
        "config": config,
        "seed": seed,
        "num_trajectories": num_trajectories,
    }

    # Save dataset if output_dir is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        # Save dataset as numpy file
        dataset_path = os.path.join(output_dir, "phase_family_dataset.npz")
        np.savez_compressed(
            dataset_path,
            trajectories=trajectories,
            frequencies=frequencies,
            config=config,
            seed=seed,
            num_trajectories=num_trajectories,
        )

        # Save metadata separately (as it contains arrays)
        metadata_path = os.path.join(output_dir, "phase_family_metadata.pkl")
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
        print(f"  - Config saved with dataset")
        print(f"  - Preview plots saved to {preview_dir}")

    return dataset


def _generate_preview_plots(dataset, output_dir, num_previews=5):
    """
    Generate preview plots for the dataset.

    Parameters
    ----------
    dataset : dict
        Dataset dictionary returned by generate_phase_family_dataset
    output_dir : str
        Directory to save preview plots
    num_previews : int
        Number of trajectories to preview
    """
    trajectories = dataset["trajectories"]
    frequencies = dataset["frequencies"]
    num_trajectories = dataset["num_trajectories"]

    # Limit number of previews
    num_previews = min(num_previews, num_trajectories)

    # Create previews for first num_previews trajectories
    for i in range(num_previews):
        traj = trajectories[i]

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
            axes[j].set_title(f"w = {freq_val:.2f}")
            axes[j].axis('off')
            plt.colorbar(im, ax=axes[j], fraction=0.046, pad=0.04)

        # Hide unused subplots
        for j in range(n_show, 6):
            axes[j].axis('off')

        fig.suptitle(f"Phase Family Trajectory {i+1}/{num_trajectories}", fontsize=14)
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
        im = axes[i].imshow(field, cmap='RdBu_r', origin='lower')
        axes[i].set_title(f"Traj {i+1}, w={frequencies[0]:.2f}")
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    fig.suptitle(f"First Frequency (w={frequencies[0]:.2f}) for {num_previews} Trajectories", fontsize=14)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "summary_first_frequency.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  Generated {num_previews + 1} preview plots in {output_dir}")