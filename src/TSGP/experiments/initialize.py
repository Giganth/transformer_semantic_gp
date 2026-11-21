"""Initialize multiple experimental runs for TSGP framework."""

import argparse
import pickle
import yaml
from copy import deepcopy
from pathlib import Path
from TSGP.config.loader import ConfigLoader

def main():
    parser = argparse.ArgumentParser(description='Initialize multiple experimental runs')
    parser.add_argument('-f', '--folder_name', 
                        help='Folder name where runs will be saved', 
                        required=True)
    parser.add_argument('-a', '--algorithm', 
                        choices=['standard_gp', 'transformer_gp', 'data_generator'],
                        help='Algorithm to use', 
                        required=False)
    parser.add_argument('-c', '--config', 
                        help='Path to configuration file (optional)',
                        default=None)
    parser.add_argument('-d', '--dataset',
                        help='Dataset to use',
                        required=False
                        )
    parser.add_argument('-n', '--num_runs', 
                        type=int,
                        help='Number of runs to prepare', 
                        required=True)
    parser.add_argument('-r', '--random_seed_start', 
                        type=int,
                        help='Starting value for random seeds',
                        required=False)
    args = parser.parse_args()

    experiment_folder = Path(args.folder_name)
    if not experiment_folder.exists():
        experiment_folder.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {experiment_folder}")
    else:
        print(f"Directory already exists: {experiment_folder}. Please use a different name.")
        return
    
    print(f"Loading configuration for algorithm: {args.algorithm}")
    base_config = ConfigLoader.load(
        config_path=args.config,
        algorithm=args.algorithm
    )

    if args.algorithm:
        base_config['algorithm'] = args.algorithm

    if args.dataset:
        base_config['problem']['dataset'] = args.dataset
    

    metadata_file = experiment_folder / 'experiment_metadata.yaml'
    with open(metadata_file, 'w') as f:
        yaml.dump(base_config, f, default_flow_style=False, sort_keys=False)
    print(f"Saved experiment metadata to {metadata_file}")
    
    for run_id in range(args.num_runs):
        
        # Deep copy base config to avoid shared mutable state between runs
        run_config = deepcopy(base_config)
        
        run_config['run_id'] = run_id
        run_config['random_seed'] = args.random_seed_start + run_id
        
        run_config['monitor_folder'] = str(experiment_folder)
        run_config['monitor_path'] = str(experiment_folder / f"monitor_run_{run_config['random_seed']}.csv")

        if run_config['algorithm'] == 'data_generator':
            run_config['data_generator']['archive_folder'] = str(experiment_folder)
            run_config['data_generator']['archive_path'] = str(experiment_folder / f"behavior_archive_{run_config['random_seed']}.pkl")

        run_data = {
            'config': run_config,
            'run_id': run_id,
            'algorithm': args.algorithm,
            'dataset': run_config['problem']['dataset'],
            'status': 'pending'
        }
        
        run_file = experiment_folder / f"run_{run_id}.pkl"
        with open(run_file, 'wb') as f:
            pickle.dump(run_data, f)
    
    print(f"\n{'='*80}")
    print(f"INITIALIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total runs created: {args.num_runs}")
    print(f"Experiment folder: {experiment_folder}")
    print(f"\nTo execute runs, use:")
    print(f"./venv/bin/python src/TSGP/experiments/run.py -f {experiment_folder} -n <num_runs_to_execute>")

if __name__ == "__main__":
    main()

