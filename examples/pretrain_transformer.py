#!/usr/bin/env python3
"""
Pretrain Transformer model for TSGP.

Example usage:
    python examples/pretrain_transformer.py --config examples/pretraining.yaml
"""

import argparse

from TSGP.config.loader import ConfigLoader
from TSGP.transformer.pretraining import TransformerPretrainer


def main():
    parser = argparse.ArgumentParser(
        description="Pretrain Transformer model for symbolic regression"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="examples/pretraining.yaml",
        help="Path to pretraining configuration file (default: examples/pretraining.yaml)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = ConfigLoader.load(config_path=args.config,algorithm="transformer_gp")
    
    # Create pretrainer
    pretrainer = TransformerPretrainer(config)
    
    # Run pretraining
    history = pretrainer.run()
    
    print("\nPretraining completed successfully!")
    print(f"Final training loss: {history.history['loss'][-1]:.6f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.6f}")


if __name__ == "__main__":
    main()
