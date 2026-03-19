"""
Command-line tool to generate figures from benchmark results.
"""
import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.report.visualization import generate_all_plots, compare_experiments


def main():
    parser = argparse.ArgumentParser(
        description='Generate visualization figures from benchmark results.'
    )
    parser.add_argument(
        '--results', '-r',
        type=str,
        required=True,
        help='Path to benchmark_results.pkl file or directory containing it'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory for figures (default: figures/ in results directory)'
    )
    parser.add_argument(
        '--compare',
        nargs='+',
        help='List of result paths to compare (multiple experiments)'
    )
    parser.add_argument(
        '--names',
        nargs='+',
        help='Names for experiments when using --compare'
    )

    args = parser.parse_args()

    # Single experiment mode
    if args.compare is None:
        # Check if path is a directory
        if os.path.isdir(args.results):
            # Look for benchmark_results.pkl in directory
            pkl_path = os.path.join(args.results, 'benchmark_results.pkl')
            if not os.path.exists(pkl_path):
                # Try to find any .pkl file
                pkl_files = list(Path(args.results).glob('*.pkl'))
                if not pkl_files:
                    raise FileNotFoundError(f"No .pkl files found in {args.results}")
                pkl_path = str(pkl_files[0])
        else:
            pkl_path = args.results

        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Results file not found: {pkl_path}")

        print(f"Generating figures from: {pkl_path}")
        df = generate_all_plots(pkl_path, args.output)
        print(f"Visualization complete. Data summary:")
        print(df.groupby(['demo', 'representation', 'metric'])[['mean', 'std']].head())

    # Multi-experiment comparison mode
    else:
        if args.names is None or len(args.names) != len(args.compare):
            # Generate default names
            args.names = [f"Experiment_{i+1}" for i in range(len(args.compare))]

        # Check all files exist
        for path in args.compare:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Results file not found: {path}")

        print(f"Comparing {len(args.compare)} experiments:")
        for name, path in zip(args.names, args.compare):
            print(f"  {name}: {path}")

        # Determine output directory
        if args.output is None:
            args.output = os.path.join(os.path.dirname(args.compare[0]), 'comparison_figures')

        compare_experiments(args.compare, args.names, args.output)
        print(f"Comparison figures saved to: {args.output}")


if __name__ == '__main__':
    main()