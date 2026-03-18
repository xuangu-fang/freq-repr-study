"""
Helmholtz equation demo: ∇²u + k²u = f(x,y) on a periodic 2D box.
Implemented with small imaginary part to avoid resonance singularities.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def generate_helmholtz_trajectory(config):
    """
    Generate a trajectory of fields for the Helmholtz equation.

    Parameters
    ----------
    config : dict
        Configuration dictionary with keys:
        - resolution: int (grid size)
        - num_frequencies: int
        - frequency_grid: dict with type, start, end
        - source: dict with type, low, high, smoothing_sigma
        - epsilon: float (optional, default=0.01) small imaginary part to avoid resonance

    Returns
    -------
    trajectory : list of ndarray
        List of 2D fields (resolution x resolution) for each k
    wave_numbers : ndarray
        Array of wave number k values
    metadata : dict
        Metadata including the source field and coordinate grids
    """
    r = config["resolution"]
    n_k = config["num_frequencies"]

    # Create k grid (wave number)
    if "frequency_grid" in config:
        freq_cfg = config["frequency_grid"]
    elif "wave_number_grid" in config:
        freq_cfg = config["wave_number_grid"]
    else:
        raise ValueError("Config must contain either 'frequency_grid' or 'wave_number_grid'")

    if freq_cfg["type"] == "linear":
        wave_numbers = np.linspace(freq_cfg["start"], freq_cfg["end"], n_k)
    elif freq_cfg["type"] == "log":
        wave_numbers = np.logspace(np.log10(freq_cfg["start"]), np.log10(freq_cfg["end"]), n_k)
    else:
        raise ValueError(f"Unsupported grid type: {freq_cfg['type']}")

    # Generate source field
    source_cfg = config["source"]
    if source_cfg["type"] == "smooth_random":
        source = _smooth_random_field(r, source_cfg)
    elif source_cfg["type"] == "gaussian_blobs":
        source = _gaussian_blobs_field(r, source_cfg)
    else:
        raise ValueError(f"Unsupported source type: {source_cfg['type']}")

    # Fourier wave numbers
    kx = np.fft.fftfreq(r, 1.0 / (2 * np.pi))
    ky = np.fft.fftfreq(r, 1.0 / (2 * np.pi))
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    k_sq = KX**2 + KY**2  # |k|²

    # Fourier transform of source
    source_hat = np.fft.fft2(source)

    # Small imaginary part to avoid resonance singularity
    epsilon = config.get("epsilon", 0.01)

    trajectory = []
    for k in wave_numbers:
        # Solve (∇² + k²) u = f in Fourier space
        # In Fourier space: ∇² -> -|k|², so equation becomes: (-|k|² + k²) u_hat = f_hat
        # Therefore: u_hat = f_hat / (k² - |k|² + iε)
        denominator = (k**2 - k_sq) + 1j * epsilon

        # Avoid division by extremely small values
        denominator = np.where(np.abs(denominator) < 1e-12, 1e-12 + 1j * epsilon, denominator)

        u_hat = source_hat / denominator
        u = np.real(np.fft.ifft2(u_hat))
        trajectory.append(u)

    metadata = {
        "source": source,
        "k_sq": k_sq,
        "epsilon": epsilon,
    }
    return trajectory, wave_numbers, metadata


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

    noise = np.random.uniform(low, high, (resolution, resolution))
    smoothed = gaussian_filter(noise, sigma=sigma, mode="wrap")
    return smoothed


def _gaussian_blobs_field(resolution, config):
    """
    Generate a field with random Gaussian blobs.

    Parameters
    ----------
    resolution : int
        Grid size
    config : dict
        Must contain keys:
        - num_blobs_min, num_blobs_max: range for number of blobs
        - sigma_min, sigma_max: range for Gaussian standard deviation
        - amplitude_min, amplitude_max: range for blob amplitudes

    Returns
    -------
    field : ndarray
        2D array of shape (resolution, resolution)
    """
    num_blobs = np.random.randint(config["num_blobs_min"], config["num_blobs_max"] + 1)

    # Create coordinate grid (normalized to [-1, 1])
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    X, Y = np.meshgrid(x, y, indexing="ij")

    field = np.zeros((resolution, resolution))

    for _ in range(num_blobs):
        # Random blob center
        cx = np.random.uniform(-0.8, 0.8)
        cy = np.random.uniform(-0.8, 0.8)

        # Random sigma and amplitude
        sigma = np.random.uniform(config["sigma_min"], config["sigma_max"])
        amplitude = np.random.uniform(config["amplitude_min"], config["amplitude_max"])

        # Gaussian blob
        r2 = (X - cx)**2 + (Y - cy)**2
        blob = amplitude * np.exp(-r2 / (2 * sigma**2))
        field += blob

    return field


def generate_helmholtz_dataset(config, num_trajectories, output_dir=None, seed=42):
    """
    Generate a dataset of Helmholtz equation trajectories.

    Parameters
    ----------
    config : dict
        Configuration dictionary for Helmholtz (same as generate_helmholtz_trajectory)
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
        - wave_numbers: ndarray of shape (num_frequencies,)
        - metadata_list: list of metadata dicts for each trajectory
        - config: input configuration
        - seed: random seed used
    """
    np.random.seed(seed)

    r = config["resolution"]
    n_k = config["num_frequencies"]

    # Create k grid (same for all trajectories)
    if "frequency_grid" in config:
        freq_cfg = config["frequency_grid"]
    elif "wave_number_grid" in config:
        freq_cfg = config["wave_number_grid"]
    else:
        raise ValueError("Config must contain either 'frequency_grid' or 'wave_number_grid'")

    if freq_cfg["type"] == "linear":
        wave_numbers = np.linspace(freq_cfg["start"], freq_cfg["end"], n_k)
    elif freq_cfg["type"] == "log":
        wave_numbers = np.logspace(np.log10(freq_cfg["start"]), np.log10(freq_cfg["end"]), n_k)
    else:
        raise ValueError(f"Unsupported frequency grid type: {freq_cfg['type']}")

    # Initialize arrays
    trajectories = np.zeros((num_trajectories, n_k, r, r), dtype=np.float32)
    metadata_list = []

    # Generate each trajectory
    for i in range(num_trajectories):
        traj, wave_numbers, meta = generate_helmholtz_trajectory(config)
        # Convert list to array and store
        trajectories[i] = np.array(traj)
        metadata_list.append(meta)

    # Create dataset dictionary
    dataset = {
        "trajectories": trajectories,      # shape (N, M, H, W)
        "wave_numbers": wave_numbers,      # shape (M,)
        "metadata_list": metadata_list,    # list of N dicts
        "config": config,
        "seed": seed,
        "num_trajectories": num_trajectories,
    }

    # Save dataset if output_dir is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        # Save dataset as numpy file
        dataset_path = os.path.join(output_dir, "helmholtz_dataset.npz")
        np.savez_compressed(
            dataset_path,
            trajectories=trajectories,
            wave_numbers=wave_numbers,
            config=config,
            seed=seed,
            num_trajectories=num_trajectories,
        )

        # Save metadata separately (as it contains arrays)
        metadata_path = os.path.join(output_dir, "helmholtz_metadata.pkl")
        import pickle
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata_list, f)

        # Generate preview plots
        preview_dir = os.path.join(output_dir, "preview")
        os.makedirs(preview_dir, exist_ok=True)
        _generate_preview_plots(dataset, preview_dir, num_previews=5)

        print(f"Dataset saved to {output_dir}")
        print(f"  - Trajectories shape: {trajectories.shape}")
        print(f"  - Wave numbers: {wave_numbers}")
        print(f"  - Config saved with dataset")
        print(f"  - Preview plots saved to {preview_dir}")

    return dataset


def _generate_preview_plots(dataset, output_dir, num_previews=5):
    """
    Generate preview plots for the Helmholtz dataset.

    Parameters
    ----------
    dataset : dict
        Dataset dictionary returned by generate_helmholtz_dataset
    output_dir : str
        Directory to save preview plots
    num_previews : int
        Number of trajectories to preview
    """
    trajectories = dataset["trajectories"]
    wave_numbers = dataset["wave_numbers"]
    num_trajectories = dataset["num_trajectories"]

    # Limit number of previews
    num_previews = min(num_previews, num_trajectories)

    # Create previews for first num_previews trajectories
    for i in range(num_previews):
        traj = trajectories[i]

        # Create figure with 2x3 subplots (show 6 k points)
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()

        # Show 6 k points (or all if less than 6)
        n_show = min(6, len(wave_numbers))
        step = max(1, len(wave_numbers) // n_show)

        for j, ax_idx in enumerate(range(0, len(wave_numbers), step)):
            if j >= n_show:
                break

            k_idx = ax_idx
            field = traj[k_idx]
            k_val = wave_numbers[k_idx]

            im = axes[j].imshow(field, cmap='RdBu_r', origin='lower')
            axes[j].set_title(f"k = {k_val:.2f}")
            axes[j].axis('off')
            plt.colorbar(im, ax=axes[j], fraction=0.046, pad=0.04)

        # Hide unused subplots
        for j in range(n_show, 6):
            axes[j].axis('off')

        fig.suptitle(f"Helmholtz Trajectory {i+1}/{num_trajectories}", fontsize=14)
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
        field = trajectories[i, 0]  # First k
        im = axes[i].imshow(field, cmap='RdBu_r', origin='lower')
        axes[i].set_title(f"Traj {i+1}, k={wave_numbers[0]:.2f}")
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    fig.suptitle(f"First Wave Number (k={wave_numbers[0]:.2f}) for {num_previews} Trajectories", fontsize=14)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "summary_first_wavenumber.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  Generated {num_previews + 1} preview plots in {output_dir}")