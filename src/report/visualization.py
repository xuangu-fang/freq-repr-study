"""
Visualization module for generating plots from benchmark results.
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List, Tuple, Optional
import pandas as pd

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.titlesize'] = 12
matplotlib.rcParams['axes.labelsize'] = 11


def load_results(result_path: str) -> Dict:
    """
    Load benchmark results from pickle file.

    Parameters
    ----------
    result_path : str
        Path to benchmark_results.pkl file.

    Returns
    -------
    dict
        Loaded results dictionary.
    """
    with open(result_path, 'rb') as f:
        results = pickle.load(f)
    return results


def extract_metrics_summary(results: Dict) -> pd.DataFrame:
    """
    Extract metrics summary into a pandas DataFrame.

    Parameters
    ----------
    results : dict
        Benchmark results dictionary.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: demo, representation, metric, mean, std, count
    """
    data = []

    for demo_name, demo_res in results['demo_results'].items():
        metrics_dict = demo_res['metrics']

        for repr_name, repr_metrics in metrics_dict.items():
            # Skip split-specific entries
            if repr_name.endswith(('_train', '_val', '_test')):
                continue

            for metric_name, metric_values in repr_metrics.items():
                # Skip split-specific entries
                if metric_name.endswith(('_train', '_val', '_test')):
                    continue

                # Ensure metric_values is a list
                if isinstance(metric_values, list) and metric_values:
                    # Flatten if needed (for array-valued metrics)
                    flat_values = []
                    for v in metric_values:
                        if isinstance(v, (np.ndarray, list)):
                            flat_values.extend(np.array(v).flatten())
                        else:
                            flat_values.append(v)

                    if flat_values:
                        mean_val = np.mean(flat_values)
                        std_val = np.std(flat_values)
                        count = len(flat_values)

                        data.append({
                            'demo': demo_name,
                            'representation': repr_name,
                            'metric': metric_name,
                            'mean': mean_val,
                            'std': std_val,
                            'count': count
                        })

    return pd.DataFrame(data)


def plot_representation_comparison(df: pd.DataFrame, output_dir: str):
    """
    Create bar plots comparing representations for each metric and demo.

    Parameters
    ----------
    df : pd.DataFrame
        Metrics summary DataFrame.
    output_dir : str
        Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Define consistent color scheme for representations
    repr_colors = {
        'raw': '#1f77b4',       # blue
        'fourier': '#ff7f0e',   # orange
        'amplitude_phase': '#2ca02c',  # green
        'pca': '#d62728',       # red
    }

    # Group by demo and metric
    for demo_name in df['demo'].unique():
        demo_df = df[df['demo'] == demo_name]

        for metric_name in demo_df['metric'].unique():
            metric_df = demo_df[demo_df['metric'] == metric_name]

            # Sort representations by mean value (ascending for "low better" metrics)
            metric_df = metric_df.sort_values('mean')
            representations = metric_df['representation'].tolist()
            means = metric_df['mean'].tolist()
            stds = metric_df['std'].tolist()

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))

            # Bar positions
            x_pos = np.arange(len(representations))

            # Colors
            colors = [repr_colors.get(repr, '#7f7f7f') for repr in representations]

            # Plot bars
            bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.8)

            # Add value labels on top of bars
            for i, (mean, std) in enumerate(zip(means, stds)):
                ax.text(i, mean + std + 0.02 * max(means),
                       f'{mean:.3g}\n±{std:.2g}',
                       ha='center', va='bottom', fontsize=9)

            # Customize plot
            ax.set_xlabel('Representation')
            ax.set_ylabel(f'{metric_name.replace("_", " ").title()}')
            ax.set_title(f'{demo_name} - {metric_name.replace("_", " ").title()} Comparison')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(representations, rotation=45, ha='right')

            # Add horizontal grid
            ax.yaxis.grid(True, linestyle='--', alpha=0.7)

            # Add "low better" note for relevant metrics
            if metric_name in ['local_smoothness', 'curvature', 'intrinsic_rank', 'interpolation_error']:
                ax.text(0.02, 0.98, 'Lower is better',
                       transform=ax.transAxes, fontsize=9,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.tight_layout()

            # Save figure
            safe_metric_name = metric_name.replace('_', '-')
            safe_demo_name = demo_name.replace('_', '-')
            filename = f'{safe_demo_name}_{safe_metric_name}_comparison.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"Saved: {filepath}")


def plot_demo_comparison(df: pd.DataFrame, output_dir: str):
    """
    Create bar plots comparing demos for each representation and metric.

    Parameters
    ----------
    df : pd.DataFrame
        Metrics summary DataFrame.
    output_dir : str
        Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Define consistent color scheme for demos
    demo_colors = {
        'phase_family': '#1f77b4',
        'screened_poisson': '#ff7f0e',
        'helmholtz': '#2ca02c',
        'wave_equation': '#d62728',
    }

    # Group by representation and metric
    for repr_name in df['representation'].unique():
        repr_df = df[df['representation'] == repr_name]

        for metric_name in repr_df['metric'].unique():
            metric_df = repr_df[repr_df['metric'] == metric_name]

            # Sort demos by mean value
            metric_df = metric_df.sort_values('mean')
            demos = metric_df['demo'].tolist()
            means = metric_df['mean'].tolist()
            stds = metric_df['std'].tolist()

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))

            # Bar positions
            x_pos = np.arange(len(demos))

            # Colors
            colors = [demo_colors.get(demo, '#7f7f7f') for demo in demos]

            # Plot bars
            bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.8)

            # Add value labels on top of bars
            for i, (mean, std) in enumerate(zip(means, stds)):
                ax.text(i, mean + std + 0.02 * max(means),
                       f'{mean:.3g}\n±{std:.2g}',
                       ha='center', va='bottom', fontsize=9)

            # Customize plot
            ax.set_xlabel('Demo Family')
            ax.set_ylabel(f'{metric_name.replace("_", " ").title()}')
            ax.set_title(f'{repr_name} Representation - {metric_name.replace("_", " ").title()} Across Demos')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(demos, rotation=45, ha='right')

            # Add horizontal grid
            ax.yaxis.grid(True, linestyle='--', alpha=0.7)

            # Add "low better" note
            if metric_name in ['local_smoothness', 'curvature', 'intrinsic_rank', 'interpolation_error']:
                ax.text(0.02, 0.98, 'Lower is better',
                       transform=ax.transAxes, fontsize=9,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.tight_layout()

            # Save figure
            safe_metric_name = metric_name.replace('_', '-')
            safe_repr_name = repr_name.replace('_', '-')
            filename = f'{safe_repr_name}_{safe_metric_name}_demo_comparison.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"Saved: {filepath}")


def plot_ranking_heatmap(df: pd.DataFrame, output_dir: str):
    """
    Create heatmap showing representation rankings across demos and metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Metrics summary DataFrame.
    output_dir : str
        Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Pivot table: mean values for each demo × representation × metric
    pivot_df = df.pivot_table(
        index=['demo', 'representation'],
        columns='metric',
        values='mean',
        aggfunc='first'
    ).reset_index()

    # For each demo and metric, compute ranking (1 = best, 4 = worst)
    rankings = []
    for demo_name in pivot_df['demo'].unique():
        demo_subset = pivot_df[pivot_df['demo'] == demo_name]

        for metric_name in ['local_smoothness', 'curvature', 'intrinsic_rank', 'interpolation_error']:
            if metric_name in demo_subset.columns:
                # Sort by value (lower is better)
                sorted_reprs = demo_subset.sort_values(metric_name)['representation'].tolist()

                for rank, repr_name in enumerate(sorted_reprs, 1):
                    rankings.append({
                        'demo': demo_name,
                        'representation': repr_name,
                        'metric': metric_name,
                        'rank': rank,
                        'value': demo_subset[demo_subset['representation'] == repr_name][metric_name].iloc[0]
                    })

    rank_df = pd.DataFrame(rankings)

    # Create heatmap for each metric
    for metric_name in rank_df['metric'].unique():
        metric_rank_df = rank_df[rank_df['metric'] == metric_name]

        # Pivot for heatmap
        heatmap_data = metric_rank_df.pivot(
            index='representation',
            columns='demo',
            values='rank'
        )

        # Sort representations by average rank
        heatmap_data['avg_rank'] = heatmap_data.mean(axis=1)
        heatmap_data = heatmap_data.sort_values('avg_rank').drop('avg_rank', axis=1)

        # Sort demos
        demos_sorted = heatmap_data.columns.tolist()

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create heatmap
        im = ax.imshow(heatmap_data.values, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=4)

        # Add text annotations
        for i in range(len(heatmap_data)):
            for j in range(len(heatmap_data.columns)):
                rank = heatmap_data.iloc[i, j]
                value = metric_rank_df[
                    (metric_rank_df['representation'] == heatmap_data.index[i]) &
                    (metric_rank_df['demo'] == heatmap_data.columns[j])
                ]['value'].iloc[0]

                ax.text(j, i, f'{int(rank)}\n({value:.2g})',
                       ha='center', va='center', fontsize=9,
                       color='white' if rank > 2.5 else 'black')

        # Customize axes
        ax.set_xticks(range(len(demos_sorted)))
        ax.set_xticklabels(demos_sorted, rotation=45, ha='right')
        ax.set_yticks(range(len(heatmap_data)))
        ax.set_yticklabels(heatmap_data.index)

        ax.set_xlabel('Demo Family')
        ax.set_ylabel('Representation')
        ax.set_title(f'{metric_name.replace("_", " ").title()} - Representation Rankings (Lower Rank = Better)')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Rank (1=Best, 4=Worst)')

        plt.tight_layout()

        # Save figure
        safe_metric_name = metric_name.replace('_', '-')
        filename = f'ranking_heatmap_{safe_metric_name}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved: {filepath}")


def plot_trajectory_visualization(results: Dict, output_dir: str,
                                 demo_name: str = None,
                                 trajectory_idx: int = 0,
                                 num_frequencies: int = 5):
    """
    Visualize a sample trajectory across frequencies.

    Parameters
    ----------
    results : dict
        Benchmark results dictionary.
    output_dir : str
        Directory to save plots.
    demo_name : str, optional
        Specific demo to visualize. If None, uses first demo.
    trajectory_idx : int
        Index of trajectory to visualize.
    num_frequencies : int
        Number of frequency samples to show.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Select demo
    if demo_name is None:
        demo_name = list(results['demo_results'].keys())[0]

    demo_res = results['demo_results'][demo_name]

    # Get raw trajectory and frequencies
    raw_trajectory = demo_res['trajectories']['raw'][trajectory_idx]
    frequencies = demo_res['trajectories']['frequencies'][trajectory_idx]

    # Select frequency indices to display
    if len(frequencies) > num_frequencies:
        step = len(frequencies) // num_frequencies
        freq_indices = list(range(0, len(frequencies), step))[:num_frequencies]
    else:
        freq_indices = list(range(len(frequencies)))

    # Create figure
    fig, axes = plt.subplots(1, len(freq_indices), figsize=(4 * len(freq_indices), 4))
    if len(freq_indices) == 1:
        axes = [axes]

    # Plot each frequency
    for i, (ax, idx) in enumerate(zip(axes, freq_indices)):
        field = raw_trajectory[idx]
        freq = frequencies[idx]

        im = ax.imshow(field, cmap='RdBu_r', origin='lower')
        ax.set_title(f'f = {freq:.2f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xticks([])
        ax.set_yticks([])

        # Add colorbar for first subplot
        if i == 0:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f'{demo_name} - Trajectory {trajectory_idx} Sample Fields', fontsize=14)
    plt.tight_layout()

    # Save figure
    safe_demo_name = demo_name.replace('_', '-')
    filename = f'{safe_demo_name}_trajectory_{trajectory_idx}_fields.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {filepath}")


def plot_metric_distributions(df: pd.DataFrame, output_dir: str):
    """
    Create box plots showing metric distributions across representations.

    Parameters
    ----------
    df : pd.DataFrame
        Metrics summary DataFrame.
    output_dir : str
        Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Note: This function requires access to raw per-trajectory data
    # Currently implemented using aggregated means/stds
    # For full implementation, need to load raw per-trajectory data

    print("Note: Distribution plots require raw per-trajectory data.")
    print("To implement fully, modify to load detailed results.")


def generate_all_plots(results_path: str, output_dir: str = None):
    """
    Generate all visualization plots from benchmark results.

    Parameters
    ----------
    results_path : str
        Path to benchmark_results.pkl file.
    output_dir : str, optional
        Directory to save plots. If None, uses same directory as results.
    """
    # Load results
    results = load_results(results_path)

    # Determine output directory
    if output_dir is None:
        results_dir = os.path.dirname(results_path)
        output_dir = os.path.join(results_dir, 'figures')

    # Create main figures directory
    os.makedirs(output_dir, exist_ok=True)

    # Extract metrics summary
    df = extract_metrics_summary(results)

    # Generate plots
    print("Generating representation comparison plots...")
    repr_comparison_dir = os.path.join(output_dir, 'representation_comparison')
    plot_representation_comparison(df, repr_comparison_dir)

    print("Generating demo comparison plots...")
    demo_comparison_dir = os.path.join(output_dir, 'demo_comparison')
    plot_demo_comparison(df, demo_comparison_dir)

    print("Generating ranking heatmaps...")
    ranking_dir = os.path.join(output_dir, 'rankings')
    plot_ranking_heatmap(df, ranking_dir)

    print("Generating trajectory visualizations...")
    trajectory_dir = os.path.join(output_dir, 'trajectories')
    plot_trajectory_visualization(results, trajectory_dir)

    print(f"All plots saved to: {output_dir}")

    # Return DataFrame for further analysis
    return df


def compare_experiments(result_paths: List[str], experiment_names: List[str],
                       output_dir: str):
    """
    Compare results from multiple experiments.

    Parameters
    ----------
    result_paths : List[str]
        List of paths to benchmark_results.pkl files.
    experiment_names : List[str]
        Names for each experiment.
    output_dir : str
        Directory to save comparison plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load and combine results
    all_data = []
    for path, name in zip(result_paths, experiment_names):
        results = load_results(path)
        df = extract_metrics_summary(results)
        df['experiment'] = name
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)

    # Create comparison plots for each metric and representation
    for metric_name in combined_df['metric'].unique():
        metric_df = combined_df[combined_df['metric'] == metric_name]

        for repr_name in metric_df['representation'].unique():
            repr_metric_df = metric_df[metric_df['representation'] == repr_name]

            # Pivot for grouped bar plot
            pivot_df = repr_metric_df.pivot_table(
                index='experiment',
                columns='demo',
                values='mean',
                aggfunc='first'
            )

            # Create grouped bar plot
            fig, ax = plt.subplots(figsize=(12, 6))

            # Bar positions
            n_experiments = len(pivot_df)
            n_demos = len(pivot_df.columns)
            bar_width = 0.8 / n_demos
            x_pos = np.arange(n_experiments)

            # Plot bars for each demo
            for i, demo_name in enumerate(pivot_df.columns):
                demo_values = pivot_df[demo_name].values
                demo_stds = repr_metric_df[repr_metric_df['demo'] == demo_name]['std'].values

                offset = (i - n_demos/2 + 0.5) * bar_width
                bars = ax.bar(x_pos + offset, demo_values, bar_width,
                            label=demo_name, alpha=0.8)

                # Add error bars
                ax.errorbar(x_pos + offset, demo_values, yerr=demo_stds,
                          fmt='none', color='black', capsize=3)

            ax.set_xlabel('Experiment')
            ax.set_ylabel(f'{metric_name.replace("_", " ").title()}')
            ax.set_title(f'{repr_name} - {metric_name.replace("_", " ").title()} Across Experiments')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(pivot_df.index, rotation=45, ha='right')
            ax.legend(title='Demo Family')
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)

            plt.tight_layout()

            # Save figure
            safe_metric_name = metric_name.replace('_', '-')
            safe_repr_name = repr_name.replace('_', '-')
            filename = f'experiment_comparison_{safe_repr_name}_{safe_metric_name}.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"Saved: {filepath}")