# Transformer Model Building Guide

This guide walks you through the complete process of building a pretrained Transformer model for Transformer Semantic Genetic Programming (TSGP). The process consists of three main steps:

1. **Data Generation** - Generate training data using evolutionary search
2. **Semantic Similarity Search** - Build similarity mappings between expressions
3. **Pretraining** - Train the Transformer model on the generated data

---

## Why the Transformer Gets Stuck on Non-Linear Problems

Before configuring data generation it is worth understanding why more training
data from linear problems does *not* solve the local-optima issue on non-linear
datasets.

### Root causes

1. **Training-distribution mismatch.**  The default `make_regression` mode
   creates purely linear target landscapes.  The GP runs that generate training
   data therefore evolve mostly linear or weakly non-linear expressions, and the
   transformer learns a semantic-space topology that reflects *those* families.
   When it is deployed on a `sin`/`exp`/polynomial target, the semantic
   distances it generates and the mutations it proposes are miscalibrated for
   the true landscape.

2. **`target_semantic_distance` is not transferable across landscapes.**  The
   conditioning signal that controls exploration vs. exploitation (set via
   `variation.transformer.target_semantic_distance`) was learned relative to
   the distribution of semantic vectors in the training archives.  On a non-
   linear problem those vectors live in a very different region of output-space,
   so a distance of 1.0 that causes useful exploration on a linear problem may
   cause either micro-steps or wild jumps on a trigonometric one.

3. **Semantic diversity of the archive drives transformer generalisation.**  The
   transformer is trained on (parent → child) pairs selected by semantic
   distance.  If every run in the archive saw a linear target, the transformer
   never learns meaningful transition paths through non-linear semantic
   neighbourhoods, so it cannot generate helpful offspring there at inference
   time.

4. **GP search landscape ruggedness.**  Non-linear targets have many more local
   optima than linear ones.  Standard variation operators (crossover, point
   mutation) are topology-blind; the transformer is supposed to escape optima by
   proposing semantically-controlled jumps.  But it can only do so reliably if
   it saw analogous non-linear topology during training.

### What to do

- **Diversify the target functions** used when generating training archives.
  Each GP run should ideally evolve on a *different* non-linear landscape so
  that the combined archive covers a wide range of semantic topologies.  The
  `target_function_pool` feature (see §1.1 below) automates this.
- **Match `input_range` to realistic problem domains** so that the semantic
  vectors in the training archive are representative of production inputs.
- **Reduce noise** for non-linear targets (e.g. `noise_level: 0.5` instead of
  `1.0`); high noise blurs the non-linear structure that the GP — and the
  transformer — needs to learn from.
- **Tune `target_semantic_distance`** per problem family: increase it for
  rugged landscapes to encourage wider exploration, decrease it once the
  transformer is tracking the attractor basin.

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
  input_range: [-2.0, 2.0]            # uniform sampling range for inputs
  num_features: 2
  num_samples_dataset: 1000
  noise_level: 0.1
```

#### Using a function pool for automatic diversity (recommended)

Running many independent GP jobs with a **fixed** target function still limits
the semantic diversity of the combined archive — all runs cluster around the
same attractor.  The `target_function_pool` key solves this: set it to a list
of expressions and each run will randomly select one (seeded by the run's
`random_seed`), so 50 parallel runs automatically cover 50 landscapes with no
extra manual configuration.

```yaml
data_generator:
  dataset_generation: target_function
  input_range: [-2.0, 2.0]
  target_function_pool:
    # polynomial families
    - "x0**3 - 2*x1**2 + x0*x1 + x2 - x3*x4"
    - "x0**2 + x1**2 - x0*x1 + x2**2 - x3 + x4"
    - "x0**4 - 3*x0**2*x1 + x1**3 - x2*x3 + x4**2"
    # trigonometric
    - "sin(x0) + cos(x1) + x2 - x3*x4"
    - "sin(x0 * x1) + cos(x2) - x3 + x4"
    - "sin(x0**2 + x1**2) + x2 - x3*x4"
    # exponential / decay
    - "exp(-x0**2) * cos(x1) + x2 - x3 + x4"
    - "log(abs(x0) + 1) + sqrt(abs(x1)) + x2 - x3*x4"
    # mixed
    - "sin(x0) + x1**2 - x0*x1 + x2*x3 - x4"
    - "x0**2 * sin(x1) + x2**2 * cos(x3) - x4"
  num_features: 5
  num_samples_dataset: 1000
  noise_level: 0.5
```

The pool covers five structural families — polynomials, trigonometric, exponential/decay, mixed, and rational-like — that create qualitatively different fitness landscapes.  Including all of them in the training archive is what teaches the transformer to navigate non-linear problems at inference time.

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