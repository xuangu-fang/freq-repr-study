from src.experiments.run_benchmark import run_benchmark
import argparse
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run_benchmark(config)

if __name__ == "__main__":
    main()