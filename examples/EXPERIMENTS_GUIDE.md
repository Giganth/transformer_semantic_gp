# TSGP Framework - Usage Guide

## Two Modes of Operation

### Mode 1: Single Run (Quick Testing)

For quick testing and development, use the `symbolic_regression.py` script:

```bash
# Run with minimal Settings
./venv/bin/python  examples/symbolic_regression.py --config examples/minimal_settings.yaml

# Run standard GP
./venv/bin/python examples/symbolic_regression.py --algorithm standard_gp

# Run transformer GP
./venv/bin/python examples/symbolic_regression.py --algorithm transformer_gp

# Use custom config
./venv/bin/python examples/symbolic_regression.py --config examples/standard_gp.yaml

# Use custom config
./venv/bin/python examples/symbolic_regression.py --config examples/transformer_gp.yaml

# Specify dataset
./venv/bin/python examples/symbolic_regression.py --algorithm transformer_gp --dataset feynman_I_27_6
```

### Mode 2: Multiple Experimental Runs

#### Step 1: Initialize Experiments

Creates experiment folder with configuration files for n independent runs.

```bash
# Initialize 30 runs of TSGP (30 runs)
./venv/bin/python src/TSGP/experiments/initialize.py \
    -f experiments_data/TSGP_1_feynman_I_43_43  \
    -a transformer_gp \
    -d feynman_I_43_43  \
    -n 30 \
    -r 1000 \
    -c examples/transformer_gp.yaml


**Arguments:**
- `-f, --folder_name`: Output folder for experiment files
- `-a, --algorithm`: `standard_gp` or `transformer_gp`
- `-d, --dataset`: Dataset name (see `src/TSGP/symbolic_regression/datasets/`)
- `-n, --num_runs`: Number of independent runs
- `-r, --random_seed_start`: Starting seed (run i uses seed r+i)
- `-c, --config`: Base config file (optional)
```
**Output:** Creates folder with `run_0.yaml`, `run_1.yaml`, ..., `run_N.yaml`

#### Step 2: Execute Runs

**Single machine:**
```bash
# Execute all runs
./venv/bin/python src/TSGP/experiments/run.py \
    -f experiments_data/test_tsgp \
    -n 10

# Execute subset of runs
./venv/bin/python src/TSGP/experiments/run.py \
    -f experiments_data/test_tsgp \
    -n 5
```

**Parallel execution (recommended):**

Using taskspooler (tsp) for parallel execution:
```bash
# Queue 30 parallel workers (each executes 1 run)
for i in {1..30}; do 
    tsp ./venv/bin/python src/TSGP/experiments/run.py \
        -f experiments_data/TSGP_1_feynman_I_43_43 \
        -n 1
done

# Set max parallel jobs (default: 1)
tsp -S 3  # Run 3 jobs simultaneously

# Monitor queue
tsp        # List all jobs
tsp -t 0   # Show output of job ID 0
tsp -C     # Clear finished jobs
tsp -r     # removes last job from queue
```

The runner script uses lock files to prevent conflicts - multiple runners can safely execute runs from the same folder.

#### Step 3: Analyze Results

After runs complete, analyze and visualize results:

```bash
# Analyze single experiment
./venv/bin/python src/TSGP/experiments/analyze.py \
    -f experiments_data/StdGP_feynman_I_43_43 \
    -o experiments_data/results/StdGP_feynman_I_43_43.pdf \
    --title "StdGP on Feynman I 43 43"

# Compare multiple experiments
./venv/bin/python src/TSGP/experiments/analyze.py \
    -f experiments_data/StdGP_feynman_I_43_43  \
    experiments_data/TSGP_01_feynman_I_43_43 \
    experiments_data/TSGP_1_feynman_I_43_43 \
    -n "Standard GP" "TSGP 01" "TSGP 1" \
    -o experiments_data/results/comparison_feynman_I_43_43.pdf \
    --title "GP vs TSGP on feynman_I_43_43 "
```

**Arguments:**
- `-f, --folders`: One or more experiment folders to analyze
- `-n, --names`: Optional names for each experiment (for legend)
- `-o, --output`: Output PDF file path
- `--title`: Main title for the plots

**Output:**
- PDF with plots showing median/quartiles across runs
- Console output with final generation statistics (median fitness, test fitness, size, runtime)


