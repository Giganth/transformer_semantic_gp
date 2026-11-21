#!/usr/bin/env python3
"""
Run similarity search to generate training pairs from archived individuals.

Example usage:
    python examples/run_similarity_search.py --config examples/similarity_search.yaml
"""

import argparse

from TSGP.config.loader import ConfigLoader
from TSGP.transformer.similarity_search import SimilaritySearcher


def main():
    """Run similarity search to create training pairs."""
    parser = argparse.ArgumentParser(
        description="Run similarity search to generate training pairs from archives"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="examples/similarity_search.yaml",
        help="Path to similarity search configuration file"
    )
    
    args = parser.parse_args()
    
    print(f"Loading configuration from: {args.config}")
    config = ConfigLoader.load(config_path=args.config, algorithm="transformer_gp")

    archive_dirs = config['model']['archive_directories']
    
    for i, archive_dir in enumerate(archive_dirs):
        print(f"Archive Directory {i+1}: {archive_dir}")
        searcher = SimilaritySearcher(config)
        searcher.run(archive_dir)

if __name__ == "__main__":
    main()
