"""Execute experimental runs prepared by initialize.py."""

import sys
import os
import pickle
import glob
import time
import argparse
import gc
from pathlib import Path
from random import shuffle, random

def main():
    parser = argparse.ArgumentParser(description='Execute experimental runs')
    parser.add_argument('-f', '--folder_name', 
                        help='Folder where pickle files are saved', 
                        required=True)
    parser.add_argument('-n', '--num_runs', 
                        type=int,
                        help='Number of runs to execute by this runner', 
                        required=True)
    parser.add_argument('-o', '--console_out', 
                        type=bool,
                        default=True,
                        help='Enable console output')
    args = parser.parse_args()

    experiment_folder = Path(args.folder_name)
    
    if not experiment_folder.exists():
        print(f"Error: Experiment folder {experiment_folder} does not exist!")
        sys.exit(1)
    
    runs_completed = 0
    
    pattern = str(experiment_folder / "**" / "run_*.pkl")
    filenames = glob.glob(pattern, recursive=True)
    
    # Shuffle to avoid conflicts when multiple runners are active
    shuffle(filenames)
    
    if args.console_out:
        print(f"Found {len(filenames)} potential run files")
        print(f"Will execute up to {args.num_runs} runs")
    
    for filename in filenames:
        if runs_completed >= args.num_runs:
            break
        
        filename_path = Path(filename)
        
        if "metadata" in filename_path.name or "monitor" in filename_path.name:
            continue

        if not filename_path.exists():
            continue
                
        lock_file = Path(str(filename_path) + ".lock")
        if lock_file.exists():
            if args.console_out:
                print(f"Skipping locked file: {filename_path.name}")
            continue
        
        try:
            lock_file.touch()
            if args.console_out:
                print(f"\n{'='*80}")
                print(f"Processing: {filename_path}")
                print(f"{'='*80}")
        except Exception as e:
            if args.console_out:
                print(f"Couldn't lock file {filename_path}: {e}")
            continue
        
        sys.stdout.flush()
        
        try:
            with open(filename_path, 'rb') as f:
                run_data = pickle.load(f)
            
            config = run_data['config']
            algorithm = run_data['algorithm']
            run_id = run_data['run_id']
            dataset = run_data['dataset']

            if args.console_out:
                print(f"Run ID: {run_id}")
                print(f"Algorithm: {algorithm}")
                print(f"Dataset: {dataset}")
                print(f"Random Seed: {config.get('random_seed')}")
            
            start_time = time.time()
            
            if algorithm == 'standard_gp':
                from TSGP.algorithms.standard_gp import StandardGP
                algo_instance = StandardGP(config)
            elif algorithm == 'transformer_gp':
                from TSGP.algorithms.ts_gp import TransformerGP
                algo_instance = TransformerGP(config)
            elif algorithm == 'data_generator':
                from TSGP.algorithms.data_generator import DataGenerator
                algo_instance = DataGenerator(config)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            if args.console_out:
                print(f"\nStarting evolutionary search...")
            
            best_solution = algo_instance.run()
            
            elapsed_time = time.time() - start_time
            
            if args.console_out:
                print(f"\n{'='*80}")
                print(f"RUN COMPLETED")
                print(f"{'='*80}")
                print(f"Best fitness: {best_solution.fitness.values[0]:.6f}")
                print(f"Best solution: {str(best_solution)}")
                print(f"Time elapsed: {elapsed_time:.2f} seconds")
            
            runs_completed += 1
            
            try:
                if args.console_out:
                    print(f"Removing job file: {filename_path}")
                filename_path.unlink()
                
                if args.console_out:
                    print(f"Removing lock file: {lock_file}")
                lock_file.unlink()
                
                del algo_instance
                del run_data
                gc.collect()
                
                # Small random delay to avoid conflicts
                time.sleep(random() * 2)
                
            except Exception as e:
                if args.console_out:
                    print(f"Warning: Couldn't remove files: {e}")
        
        except Exception as e:
            if args.console_out:
                print(f"\n{'!'*80}")
                print(f"ERROR processing run")
                print(f"{'!'*80}")
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
            
            error_file = filename_path.parent / f"error_run_{run_data['run_id']}.txt"
            with open(error_file, 'w') as f:
                f.write(f"Error in run {run_data['run_id']}\n")
                f.write(f"Error: {str(e)}\n")
                import traceback
                traceback.print_exc(file=f)
            
            if lock_file.exists():
                lock_file.unlink()
            
            continue
    
    if args.console_out:
        print(f"\n{'='*80}")
        print(f"RUNNER COMPLETE")
        print(f"{'='*80}")
        print(f"Runs completed: {runs_completed}")
    
    sys.stdout.flush()

if __name__ == "__main__":
    main()
