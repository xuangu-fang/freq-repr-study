#!/usr/bin/env python
"""
Test script for Helmholtz complex field generation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from src.data_gen.helmholtz import generate_helmholtz_trajectory


def test_complex_generation():
    """Test generating complex Helmholtz fields"""
    config = {
        'resolution': 64,
        'num_frequencies': 10,
        'wave_number_grid': {
            'type': 'linear',
            'start': 0.5,
            'end': 5.0
        },
        'source': {
            'type': 'gaussian_blobs',
            'num_blobs_min': 2,
            'num_blobs_max': 5,
            'sigma_min': 0.03,
            'sigma_max': 0.12,
            'amplitude_min': 0.5,
            'amplitude_max': 1.5
        },
        'epsilon': 0.01
    }

    print("Testing real field generation (default)...")
    config_real = config.copy()
    traj_real, wave_numbers, meta_real = generate_helmholtz_trajectory(config_real)
    print(f"  Wave numbers: {wave_numbers}")
    print(f"  Trajectory length: {len(traj_real)}")
    print(f"  Field shape: {traj_real[0].shape}")
    print(f"  Field dtype: {traj_real[0].dtype}")
    print(f"  Field is complex? {np.iscomplexobj(traj_real[0])}")

    print("\nTesting complex field generation...")
    config_complex = config.copy()
    config_complex['return_complex'] = True
    traj_complex, wave_numbers, meta_complex = generate_helmholtz_trajectory(config_complex)
    print(f"  Field dtype: {traj_complex[0].dtype}")
    print(f"  Field is complex? {np.iscomplexobj(traj_complex[0])}")

    # Compare real part of complex field with real-only field
    diff = np.max(np.abs(traj_complex[0].real - traj_real[0]))
    print(f"  Max difference between real part of complex and real-only: {diff}")

    # Check that imaginary part is not negligible
    imag_max = np.max(np.abs(traj_complex[0].imag))
    print(f"  Maximum imaginary part magnitude: {imag_max}")

    # Check energy conservation: |u|^2 should be positive
    energy = np.mean(np.abs(traj_complex[0])**2)
    print(f"  Average energy (|u|^2): {energy}")

    # Visualize first field
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Real field
    im1 = axes[0, 0].imshow(traj_real[0], cmap='RdBu_r')
    axes[0, 0].set_title('Real-only field')
    plt.colorbar(im1, ax=axes[0, 0])

    # Complex field components
    im2 = axes[0, 1].imshow(traj_complex[0].real, cmap='RdBu_r')
    axes[0, 1].set_title('Complex field (real)')
    plt.colorbar(im2, ax=axes[0, 1])

    im3 = axes[0, 2].imshow(traj_complex[0].imag, cmap='RdBu_r')
    axes[0, 2].set_title('Complex field (imag)')
    plt.colorbar(im3, ax=axes[0, 2])

    im4 = axes[1, 0].imshow(np.abs(traj_complex[0]), cmap='viridis')
    axes[1, 0].set_title('Amplitude')
    plt.colorbar(im4, ax=axes[1, 0])

    im5 = axes[1, 1].imshow(np.angle(traj_complex[0]), cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axes[1, 1].set_title('Phase')
    plt.colorbar(im5, ax=axes[1, 1])

    # Difference
    im6 = axes[1, 2].imshow(traj_complex[0].real - traj_real[0], cmap='RdBu_r')
    axes[1, 2].set_title('Difference (complex.real - real)')
    plt.colorbar(im6, ax=axes[1, 2])

    plt.tight_layout()
    plt.savefig('helmholtz_complex_test.png', dpi=150)
    print("\nSaved visualization to helmholtz_complex_test.png")

    # Check variation across frequencies
    print("\nChecking variation across frequencies:")
    for i, k in enumerate(wave_numbers):
        if i == 0 or i == len(wave_numbers)-1:
            field = traj_complex[i]
            amp_mean = np.mean(np.abs(field))
            print(f"  k={k:.2f}: mean amplitude = {amp_mean:.4f}")

    # Compute relative change between first and last frequency
    first_amp = np.mean(np.abs(traj_complex[0]))
    last_amp = np.mean(np.abs(traj_complex[-1]))
    rel_change = (last_amp - first_amp) / first_amp
    print(f"  Relative amplitude change from k={wave_numbers[0]:.2f} to k={wave_numbers[-1]:.2f}: {rel_change:.2%}")

    return traj_real, traj_complex, wave_numbers


def test_representation_compatibility():
    """Test that representation modules can handle complex fields"""
    print("\n" + "="*60)
    print("Testing representation module compatibility with complex fields")
    print("="*60)

    # Generate a complex trajectory
    config = {
        'resolution': 32,  # Smaller for faster testing
        'num_frequencies': 5,
        'wave_number_grid': {'type': 'linear', 'start': 0.5, 'end': 2.0},
        'source': {'type': 'gaussian_blobs', 'num_blobs_min': 1, 'num_blobs_max': 2,
                   'sigma_min': 0.05, 'sigma_max': 0.1, 'amplitude_min': 0.5, 'amplitude_max': 1.0},
        'return_complex': True,
        'epsilon': 0.01
    }

    traj, wave_numbers, _ = generate_helmholtz_trajectory(config)
    print(f"Generated complex trajectory: {len(traj)} fields, shape {traj[0].shape}")

    # Test each representation
    from src.representations import RawRepresentation, FourierRepresentation, \
                                   AmplitudePhaseRepresentation, PCARepresentation

    representations = [
        ('Raw', RawRepresentation()),
        ('Fourier', FourierRepresentation({'keep_all': True})),
        ('AmplitudePhase', AmplitudePhaseRepresentation({'separate_channels': True})),
        ('PCA', PCARepresentation({'n_components': 0.95}))
    ]

    # Use first trajectory as training data for representations that need fitting
    train_data = traj[:2]  # Use first two fields as training

    for name, repr_module in representations:
        print(f"\n--- Testing {name} representation ---")
        try:
            # Fit if needed
            repr_module.fit(train_data)
            print(f"  Fit successful")

            # Transform
            transformed = repr_module.transform(traj)
            print(f"  Transform successful: {len(transformed)} vectors")
            if transformed:
                print(f"    First vector shape: {transformed[0].shape}")
                print(f"    First vector dtype: {transformed[0].dtype}")

            # Inverse transform (if available)
            reconstructed = repr_module.inverse_transform(transformed[:2])  # Just first two
            print(f"  Inverse transform successful: {len(reconstructed)} fields")
            if reconstructed:
                print(f"    Reconstructed shape: {reconstructed[0].shape}")
                print(f"    Reconstructed dtype: {reconstructed[0].dtype}")
                print(f"    Reconstructed is complex? {np.iscomplexobj(reconstructed[0])}")

                # Check reconstruction error
                orig = traj[0]
                recon = reconstructed[0]
                if orig.shape == recon.shape:
                    if np.iscomplexobj(orig) and np.iscomplexobj(recon):
                        error = np.mean(np.abs(orig - recon))
                        print(f"    Reconstruction error (mean abs diff): {error:.6f}")
                    else:
                        error = np.mean(np.abs(orig.real - recon))
                        print(f"    Reconstruction error (real parts): {error:.6f}")

        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    print("Helmholtz Complex Field Generation Test")
    print("="*60)

    traj_real, traj_complex, wave_numbers = test_complex_generation()
    test_representation_compatibility()

    print("\n" + "="*60)
    print("Test completed successfully!")
    print("Next steps:")
    print("1. Run an experiment with configs/demos/helmholtz_low_freq.yaml")
    print("2. Generate GIFs using: python -m src.report.complex_trajectory_gif")
    print("3. Compare real vs complex field metrics")