#!/usr/bin/env python3
import argparse
from TSGP.config.loader import ConfigLoader

def main():
    parser = argparse.ArgumentParser(description="Run symbolic regression example")
    parser.add_argument(
        "--algorithm", 
        choices=["standard_gp", "transformer_gp"],
        default="transformer_gp",
        help="Which algorithm to use"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom configuration file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset to use for symbolic regression"
    )
    args = parser.parse_args()

    # Load configuration (merges defaults with user config if provided)
    config = ConfigLoader.load(
        config_path=args.config,
        algorithm=args.algorithm
    )
    
    # Run the appropriate algorithm
    if config['algorithm'] == "standard_gp":
        print(f"Running Standard GP on {config['problem']['dataset']}...")
        from TSGP.algorithms.standard_gp import StandardGP

        algorithm = StandardGP(config)

    elif config['algorithm'] == "transformer_gp":
        print(f"Running Transformer Semantic GP on {config['problem']['dataset']}...")
        from TSGP.algorithms.ts_gp import TransformerGP
        
        algorithm = TransformerGP(config)
    else:
        raise ValueError(f"Unknown algorithm: {config['algorithm']}")

    # Execute the search
    best_solution = algorithm.run()
    
    # Display results
    print(f"\nBest solution found for {config['problem']['dataset']}: {best_solution}")
    print(f"Expression: {str(best_solution)}")
    print(f"Training RMSE: {best_solution.fitness.values[0]:.6f}")

if __name__ == "__main__":
    main()