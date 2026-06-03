# Transformer Model Building Guide

This guide walks you through the complete process of building a pretrained Transformer model for Transformer Semantic Genetic Programming (TSGP). The process consists of three main steps:

1. **Data Generation** - Generate training data using evolutionary search
2. **Semantic Similarity Search** - Build similarity mappings between expressions
3. **Pretraining** - Train the Transformer model on the generated data

---

## Step 1: Data Generation

The first step is to generate training data by running evolutionary algorithms that produce simplified symbolic expressions along with their semantic vectors. This data will be used to train the Transformer model.

### 1.1 Configure Data Generation

Create or modify your data generation configuration file (e.g., `examples/data_generator.yaml`).

Note that "num_features" controls the dimensionality.

#### Dataset Generation Modes

Two modes are available via the `dataset_generation` key:

**`make_regression` (default)** – generates a random linear regression dataset
using [`sklearn.datasets.make_regression`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html):

```yaml
data_generator:
  dataset_generation: make_regression
  num_features: 5
  num_samples_dataset: 1000
  noise_level: 1.0
```

**`target_function`** – generates data by evaluating a user-supplied symbolic
expression.  Standard NumPy functions (`sin`, `cos`, `exp`, `log`, `sqrt`, …)
are available and variables must be named `x0`, `x1`, … matching `num_features`:

```yaml
data_generator:
  dataset_generation: target_function
  target_function: "sin(x0) + x1**2"  # any NumPy-compatible expression
  input_range: [-1.0, 1.0]            # uniform sampling range for inputs
  num_features: 2
  num_samples_dataset: 1000
  noise_level: 0.1
```

Using `target_function` mode lets you focus the GP search (and thus the
training data) on the structural diversity around a specific target you care
about, which can improve the transformer's performance on related problems.

### 1.2 Initialize Multiple Runs

Use the `initialize.py` script to create multiple experimental runs with different random seeds. This creates pickle files for each run that will be executed in parallel.


```bash
./venv/bin/python src/TSGP/experiments/initialize.py \
    -f experiments_data/GP_Archive_5 \
    -a data_generator \
    -n 50 \
    -r 1000 \
    -c examples/data_generator.yaml
```

**Parameters:**
- `-f`: Folder name for the experiment (where pickle files will be saved)
- `-a`: Algorithm name (`data_generator`)
- `-n`: Number of runs to generate
- `-r`: Starting random seed (will increment for each run: 1000, 1001, 1002, ...)
- `-c`: Path to configuration YAML file

### 1.3 Execute Data Generation Runs

Run the experiments using the `run.py` script. 

**Single machine:**
```bash
# Execute all runs sequentially
./venv/bin/python src/TSGP/experiments/run.py \
    -f experiments_data/GP_Archive_5 \
    -n 50
```

**Parallel execution (recommended):**

Using taskspooler (tsp) for parallel execution:
```bash
# Queue 50 parallel workers (each executes 1 run)
for i in {1..50}; do 
    tsp ./venv/bin/python src/TSGP/experiments/run.py \
        -f experiments_data/GP_Archive_5 \
        -n 1
done

# Set max parallel jobs (adjust based on your CPU cores)
tsp -S 10  # Run 10 jobs simultaneously

# Monitor queue
tsp        # List all jobs
tsp -t 0   # Show output of job ID 0
tsp -C     # Clear finished jobs
```

**Parameters:**
- `-f`: Folder containing the experiment pickle files
- `-n`: Number of runs this runner should execute

The runner script uses lock files to prevent conflicts - multiple runners can safely execute runs from the same folder.

**What happens:**
- Each run evolves a population of symbolic expressions
- Expressions are simplified using SymPy
- Unique expressions and their semantic vectors are stored in an archive
- Archives are saved as pickle files in the configured output directory

### 1.4 Generate Data for Multiple Dimensionalities

To train a d-dimensional transformer model, generate archives with different dimensionalities.

## Step 2: Semantic Similarity Search

After generating archives, the next step is to find pairs of semantically similar expressions. This creates training pairs for the Transformer model.

### 2.1 Run Similarity Search

Use the `run_similarity_search.py` script to process your generated archives:

```bash
./venv/bin/python examples/run_similarity_search.py --config examples/similarity_search.yaml
```

**Parameters:**
- `--config`: Path to similarity search configuration YAML file

**What happens:**
- Loads all expressions from archive files
- Computes pairwise semantic distances (Euclidean distance)
- Selects pairs within the similarity threshold
- Saves pairs as (expression1, expression2, distance) tuples


## Step 3: Pretraining the Transformer Model

The final step is to train the Transformer model using the generated similarity pairs.

### 3.1 Configure Pretraining

Create or modify your pretraining configuration file (e.g., `examples/pretraining.yaml`).

**Important:** Update the `archive_directories` paths in your configuration to match where you saved the archives from Step 1. By default, the configuration uses `experiments_data/GP_Archive_2/3/4/5`, but if you used different paths (e.g., `experiments_data/GP_Archive_5`), you must update these paths accordingly.

### 3.2 Run Pretraining

Execute the pretraining script:

```bash
./venv/bin/python examples/pretrain_transformer.py \
    --config examples/pretraining.yaml
```

**What happens:**
- Loads all similarity pairs from specified archive directories
- Creates vocabulary from functions, variables, and ephemerals
- Builds encoder/decoder vectorization layers
- Initializes Transformer model
- Trains on (parent_expression → child_expression) mappings
- Saves best model checkpoint based on validation loss
- Logs training history to CSV