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
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = ConfigLoader.load(config_path=args.config, algorithm="transformer_gp")

    # Get archive directories from config
    archive_dirs = config['model']['archive_directories']
    
    total_pairs = 0
    for i, archive_dir in enumerate(archive_dirs):
        print(f"Archive Directory {i+1}: {archive_dir}")
        # Create output path
        output_path = f"{archive_dir}/similar_pairs.pkl"

        # Run similarity search
        searcher = SimilaritySearcher(config)
        similar_pairs = searcher.run(archive_dir, output_path)
        total_pairs += len(similar_pairs)

        print(f"\nGenerated {len(similar_pairs)} training pairs from {archive_dir}")
        for j, pair in enumerate(similar_pairs[:3]):
            print(f"  {j+1}. {pair[0][:50]}... <-> {pair[1][:50]}... (distance: {pair[2]:.4f})")
    
    print(f"\nTotal training pairs generated: {total_pairs}")

if __name__ == "__main__":
    main()
