"""
Complex field trajectory GIF generation.

Generates GIF animations for PDE trajectory visualization, supporting both real
and complex fields. For complex fields, options to display amplitude, phase,
real part, or imaginary part.

Usage example:
    python -m src.report.complex_trajectory_gif --results outputs/experiment_name \
           --demo helmholtz --trajectory_idx 0 --output outputs/gifs
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import pickle
import os
import argparse


def load_trajectory_data(results_path, demo_name, trajectory_idx=0):
    """Load trajectory data from results.pkl"""
    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    # Extract raw trajectory data
    raw_trajectories = results['trajectories']['raw']
    frequencies_list = results['trajectories']['frequencies']

    trajectory = raw_trajectories[trajectory_idx]  # shape: (num_freq, H, W)
    frequencies = frequencies_list[trajectory_idx]  # get frequencies for this trajectory

    return trajectory, frequencies


def is_complex_trajectory(trajectory):
    """Check if trajectory contains complex fields"""
    return np.iscomplexobj(trajectory[0]) if len(trajectory) > 0 else False


def create_field_gif(trajectory, frequencies, output_path, fps=5,
                     component='real', cmap='RdBu_r'):
    """
    Create GIF animation of field evolution.

    Parameters
    ----------
    trajectory : ndarray
        Shape (num_freq, H, W) or (num_freq, H, W) complex
    frequencies : ndarray
        Frequency values for each frame
    output_path : str
        Path to save GIF
    fps : int
        Frames per second
    component : str
        For complex fields: 'real', 'imag', 'amplitude', 'phase'
        For real fields: ignored
    cmap : str
        Colormap for real-valued fields
    """
    num_frames = len(frequencies)
    complex_fields = is_complex_trajectory(trajectory)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Extract first frame based on component
    if complex_fields:
        if component == 'real':
            first_frame = trajectory[0].real
            title_prefix = 'Real part'
        elif component == 'imag':
            first_frame = trajectory[0].imag
            title_prefix = 'Imag part'
        elif component == 'amplitude':
            first_frame = np.abs(trajectory[0])
            title_prefix = 'Amplitude'
            cmap = 'viridis'
        elif component == 'phase':
            first_frame = np.angle(trajectory[0])
            title_prefix = 'Phase'
            cmap = 'hsv'
        else:
            raise ValueError(f"Unknown component: {component}")
    else:
        first_frame = trajectory[0]
        title_prefix = 'Field'

    # Determine vmin/vmax for consistent scaling
    if component == 'phase':
        vmin, vmax = -np.pi, np.pi
    else:
        # Compute global min/max across all frames for consistent scaling
        if complex_fields:
            if component == 'real':
                all_vals = np.concatenate([f.real.flatten() for f in trajectory])
            elif component == 'imag':
                all_vals = np.concatenate([f.imag.flatten() for f in trajectory])
            elif component == 'amplitude':
                all_vals = np.concatenate([np.abs(f).flatten() for f in trajectory])
            else:
                all_vals = first_frame.flatten()
        else:
            all_vals = np.concatenate([f.flatten() for f in trajectory])
        vmin, vmax = np.percentile(all_vals, [1, 99])
        # Ensure symmetric colormap for real/imag parts
        if component in ['real', 'imag']:
            max_abs = max(abs(vmin), abs(vmax))
            vmin, vmax = -max_abs, max_abs

    # Initialize image
    im = ax.imshow(first_frame, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    ax.set_title(f'{title_prefix} | Frequency = {frequencies[0]:.2f}')
    fig.colorbar(im, ax=ax)

    def update(frame):
        if complex_fields:
            if component == 'real':
                frame_data = trajectory[frame].real
            elif component == 'imag':
                frame_data = trajectory[frame].imag
            elif component == 'amplitude':
                frame_data = np.abs(trajectory[frame])
            elif component == 'phase':
                frame_data = np.angle(trajectory[frame])
        else:
            frame_data = trajectory[frame]

        im.set_data(frame_data)
        ax.set_title(f'{title_prefix} | Frequency = {frequencies[frame]:.2f}')
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=num_frames,
                                  interval=1000/fps, blit=True)

    # Save GIF
    writer = PillowWriter(fps=fps)
    ani.save(output_path, writer=writer)
    plt.close(fig)
    print(f"Saved GIF to {output_path}")


def create_complex_multipanel_gif(trajectory, frequencies, output_path, fps=5):
    """
    Create multipanel GIF showing real, imag, amplitude, and phase of complex field.
    Only for complex trajectories.
    """
    if not is_complex_trajectory(trajectory):
        raise ValueError("Trajectory is not complex, cannot create multipanel GIF")

    num_frames = len(frequencies)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Initialize images for each component
    components = [
        ('Real part', trajectory[0].real, 'RdBu_r'),
        ('Imag part', trajectory[0].imag, 'RdBu_r'),
        ('Amplitude', np.abs(trajectory[0]), 'viridis'),
        ('Phase', np.angle(trajectory[0]), 'hsv')
    ]

    ims = []
    for ax, (title, data, cmap) in zip(axes, components):
        if title == '相位':
            vmin, vmax = -np.pi, np.pi
        else:
            vmin, vmax = np.percentile(data.flatten(), [1, 99])
            if title in ['Real part', 'Imag part']:
                max_abs = max(abs(vmin), abs(vmax))
                vmin, vmax = -max_abs, max_abs

        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title(title)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ims.append(im)

    fig.suptitle(f'Frequency = {frequencies[0]:.2f}', fontsize=14)
    plt.tight_layout()

    def update(frame):
        # Update each component
        ims[0].set_data(trajectory[frame].real)
        ims[1].set_data(trajectory[frame].imag)
        ims[2].set_data(np.abs(trajectory[frame]))
        ims[3].set_data(np.angle(trajectory[frame]))

        fig.suptitle(f'Frequency = {frequencies[frame]:.2f}', fontsize=14)
        return ims

    ani = animation.FuncAnimation(fig, update, frames=num_frames,
                                  interval=1000/fps, blit=True)

    writer = PillowWriter(fps=fps)
    ani.save(output_path, writer=writer)
    plt.close(fig)
    print(f"Saved multipanel GIF to {output_path}")


def create_point_trace_gif(trajectory, frequencies, point_coord, output_path, fps=5,
                          component='real'):
    """
    Create GIF showing field evolution and frequency response at a spatial point.
    """
    num_frames = len(frequencies)
    complex_fields = is_complex_trajectory(trajectory)

    # Extract point trace based on component
    if complex_fields:
        if component == 'real':
            point_trace = trajectory[:, point_coord[0], point_coord[1]].real
            title_prefix = 'Real part'
        elif component == 'imag':
            point_trace = trajectory[:, point_coord[0], point_coord[1]].imag
            title_prefix = 'Imag part'
        elif component == 'amplitude':
            point_trace = np.abs(trajectory[:, point_coord[0], point_coord[1]])
            title_prefix = 'Amplitude'
        elif component == 'phase':
            point_trace = np.angle(trajectory[:, point_coord[0], point_coord[1]])
            title_prefix = 'Phase'
        else:
            raise ValueError(f"Unknown component: {component}")
    else:
        point_trace = trajectory[:, point_coord[0], point_coord[1]]
        title_prefix = 'Field'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: field image with point marker
    if complex_fields and component in ['real', 'imag', 'amplitude', 'phase']:
        if component == 'real':
            first_field = trajectory[0].real
        elif component == 'imag':
            first_field = trajectory[0].imag
        elif component == 'amplitude':
            first_field = np.abs(trajectory[0])
        elif component == 'phase':
            first_field = np.angle(trajectory[0])
    else:
        first_field = trajectory[0]

    im = ax1.imshow(first_field, cmap='RdBu_r', origin='lower')
    ax1.plot(point_coord[1], point_coord[0], 'ro', markersize=10)
    ax1.set_title(f'{title_prefix} field (Frequency={frequencies[0]:.2f})')
    fig.colorbar(im, ax=ax1)

    # Right: frequency response curve
    line, = ax2.plot(frequencies[:1], point_trace[:1], 'b-', linewidth=2)
    current_point, = ax2.plot(frequencies[0], point_trace[0], 'ro', markersize=8)
    ax2.set_xlim(frequencies[0], frequencies[-1])
    ax2.set_ylim(point_trace.min(), point_trace.max())
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel(f'{title_prefix} value')
    ax2.set_title('Spatial point frequency response')
    ax2.grid(True)

    def update(frame):
        # Update left image
        if complex_fields and component in ['real', 'imag', 'amplitude', 'phase']:
            if component == 'real':
                field_data = trajectory[frame].real
            elif component == 'imag':
                field_data = trajectory[frame].imag
            elif component == 'amplitude':
                field_data = np.abs(trajectory[frame])
            elif component == 'phase':
                field_data = np.angle(trajectory[frame])
        else:
            field_data = trajectory[frame]

        im.set_data(field_data)
        ax1.set_title(f'{title_prefix} field (Frequency={frequencies[frame]:.2f})')

        # Update right curve
        line.set_data(frequencies[:frame+1], point_trace[:frame+1])
        current_point.set_data([frequencies[frame]], [point_trace[frame]])

        return [im, line, current_point]

    ani = animation.FuncAnimation(fig, update, frames=num_frames,
                                  interval=1000/fps, blit=True)

    writer = PillowWriter(fps=fps)
    ani.save(output_path, writer=writer)
    plt.close(fig)
    print(f"Saved point trace GIF to {output_path}")


def main():
    """Main function to generate GIFs for all PDEs"""
    parser = argparse.ArgumentParser(description='Generate PDE trajectory GIF animations')
    parser.add_argument('--results', required=True,
                       help='Path to experiment results directory')
    parser.add_argument('--demo', required=True,
                       help='Demo name (e.g., helmholtz, phase_family)')
    parser.add_argument('--trajectory_idx', type=int, default=0,
                       help='Trajectory index to visualize')
    parser.add_argument('--output', default='outputs/trajectory_gifs',
                       help='Output directory for GIFs')
    parser.add_argument('--fps', type=int, default=3,
                       help='GIF frame rate')
    parser.add_argument('--components', nargs='+',
                       default=['real', 'imag', 'amplitude', 'phase'],
                       help='Components to visualize for complex fields')
    parser.add_argument('--multipanel', action='store_true',
                       help='Generate multipanel GIF for complex fields')

    args = parser.parse_args()

    # Construct results path
    results_path = os.path.join(args.results, 'demo_results', args.demo, 'results.pkl')
    if not os.path.exists(results_path):
        # Try alternative path structure
        results_path = os.path.join(args.results, 'results.pkl')
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"Results file not found: {results_path}")

    # Load trajectory data
    trajectory, frequencies = load_trajectory_data(
        results_path, args.demo, args.trajectory_idx
    )

    # Convert trajectory from list to numpy array if needed
    if isinstance(trajectory, list):
        trajectory = np.array(trajectory)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    complex_fields = is_complex_trajectory(trajectory)

    print(f"Generating GIFs for {args.demo} (trajectory {args.trajectory_idx})")
    print(f"  Complex fields: {complex_fields}")
    print(f"  Frequencies: {len(frequencies)} points from {frequencies[0]:.2f} to {frequencies[-1]:.2f}")

    # Generate GIFs based on field type
    if complex_fields:
        # Complex field visualizations
        if args.multipanel:
            multipanel_path = os.path.join(
                args.output, f"{args.demo}_complex_multipanel.gif"
            )
            create_complex_multipanel_gif(trajectory, frequencies, multipanel_path, args.fps)

        # Individual component GIFs
        for component in args.components:
            gif_path = os.path.join(
                args.output, f"{args.demo}_{component}.gif"
            )
            create_field_gif(trajectory, frequencies, gif_path, args.fps, component)

            # Point trace GIF (center point)
            H, W = trajectory.shape[1], trajectory.shape[2]
            center_point = (H//2, W//2)
            point_gif_path = os.path.join(
                args.output, f"{args.demo}_{component}_point_trace.gif"
            )
            create_point_trace_gif(
                trajectory, frequencies, center_point, point_gif_path, args.fps, component
            )
    else:
        # Real field visualizations
        gif_path = os.path.join(args.output, f"{args.demo}_field.gif")
        create_field_gif(trajectory, frequencies, gif_path, args.fps)

        # Point trace GIF
        H, W = trajectory.shape[1], trajectory.shape[2]
        center_point = (H//2, W//2)
        point_gif_path = os.path.join(args.output, f"{args.demo}_point_trace.gif")
        create_point_trace_gif(
            trajectory, frequencies, center_point, point_gif_path, args.fps
        )

    print(f"All GIFs saved to {args.output}")


if __name__ == '__main__':
    main()