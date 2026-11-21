# Running Experiments

## Single run

`symbolic_regression.py` runs one search with the settings of a config file:

```bash
python examples/symbolic_regression.py --config examples/transformer_gp_TSGP.yaml     # TSGP
python examples/symbolic_regression.py --config examples/transformer_gp_TSGP-2.yaml   # TSGP-2
python examples/symbolic_regression.py --config examples/standard_gp.yaml             # Standard GP

# Same configuration on another problem
python examples/symbolic_regression.py --config examples/standard_gp.yaml --dataset feynman_II_13_17
```

Problems are the CSV files in `src/TSGP/symbolic_regression/datasets/`, named without
the extension. The monitor CSV is written to `experiments_data/logs/`.

## Batch runs (as in the paper)

### Step 1: Initialize

`initialize.py` writes one config per run into an experiment folder. Run `i` uses seed
`r + i`.

```bash
python src/TSGP/experiments/initialize.py \
    -f experiments_data/TSGP_feynman_I_34_27 \
    -a transformer_gp \
    -c examples/transformer_gp_TSGP.yaml \
    -d feynman_I_34_27 \
    -n 30 \
    -r 1000
```

- `-f`: experiment folder
- `-a`: `standard_gp` or `transformer_gp`
- `-c`: base config
- `-d`: problem
- `-n`: number of runs
- `-r`: first seed

Repeat it for the other problems and the other two operators
(`examples/standard_gp.yaml`, `examples/transformer_gp_TSGP-2.yaml`).

### Step 2: Execute

```bash
# Execute 30 runs of one folder in sequence
python src/TSGP/experiments/run.py -f experiments_data/TSGP_feynman_I_34_27 -n 30
```

The runner takes a lock file per run. Several runners can therefore work on the same
folder in parallel, and each picks the next run nobody has started. With
[task spooler](https://viric.name/soft/ts/):

```bash
for i in {1..30}; do
    tsp python src/TSGP/experiments/run.py -f experiments_data/TSGP_feynman_I_34_27 -n 1
done
tsp -S 3   # number of jobs running at the same time
```

Each run writes `monitor_run_<seed>.csv`, with one row per generation, into the folder.

### Step 3: Analyze

`analyze.py` aggregates the monitor logs of a folder into a PDF:

```bash
python src/TSGP/experiments/analyze.py -f experiments_data/TSGP_feynman_I_34_27 -o results.pdf
```
