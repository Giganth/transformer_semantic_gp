#!/usr/bin/env python3

"""
Example script to run the Data Generator algorithm.

It is recommended to initialize an experiment using the initialize.py script to create multiple Data Archives.

Example usage:
    python examples/run_data_generator.py --config examples/data_generator_TSGP-2.yaml
"""

import argparse
from TSGP.config.loader import ConfigLoader
from TSGP.algorithms.data_generator import DataGenerator

def main():
    parser = argparse.ArgumentParser(
        description="Run Data Generator algorithm for creating training archives"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="examples/data_generator_TSGP-2.yaml",
        help="Path to data generator configuration file"
    )
    
    args = parser.parse_args()

    print(f"Loading configuration from: {args.config}")
    config = ConfigLoader.load(config_path=args.config, algorithm="data_generator")

    print("Running Data Generator...")

    algorithm = DataGenerator(config)

    algorithm.run()


if __name__ == "__main__":
    main()