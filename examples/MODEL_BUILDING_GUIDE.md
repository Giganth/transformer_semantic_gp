# Building the Transformer Models

Both models are trained in the same three steps. They differ only in the target
functions of step 1:

| Model | Corpus | Data generation | Pretraining |
|---|---|---|---|
| TSGP | additive targets | `data_generator_TSGP.yaml` | `pretraining_TSGP.yaml` |
| TSGP-2 | add, sub, mul, div, sin, sqrt | `data_generator_TSGP-2.yaml` | `pretraining_TSGP-2.yaml` |

Steps 1 and 2 produce the training and test pairs
(`trainingdata/{train,test}_pairs_*.pkl.zip`). With them you can go straight to step 3.

## Step 1: Archives from GP runs

Each archive run draws a random target function, runs GP on it, and archives the
simplified programs together with their outputs on the semantic testset. The paper uses
100 runs (seeds 1000–1099) per corpus and number of variables d = 2, 3, 4, 5.

Set `data_generator.num_features` in the config to d, then:

```bash
python src/TSGP/experiments/initialize.py \
    -f experiments_data/GP_Archive_5_TSGP-2 \
    -a data_generator \
    -c examples/data_generator_TSGP-2.yaml \
    -n 100 \
    -r 1000

python src/TSGP/experiments/run.py -f experiments_data/GP_Archive_5_TSGP-2 -n 100
```

As with the search runs, several runners can work on one folder in parallel (see
[EXPERIMENTS_GUIDE.md](EXPERIMENTS_GUIDE.md)).

The configs record the settings of the paper's archives, so regenerating them reproduces
the training data of the published models.

## Step 2: Similar pairs

For every archive file, a FAISS index finds the nearest neighbours of each program in
semantic space. Pairs closer than `similarity_threshold` are kept and written next to the
archive as `similar_pairs_<seed>.pkl`:

```bash
python examples/run_similarity_search.py --config examples/similarity_search.yaml
```

The config lists all eight archive folders. Interrupted searches resume where they
stopped.

## Step 3: Pretraining

```bash
python examples/pretrain_transformer.py --config examples/pretraining_TSGP.yaml
python examples/pretrain_transformer.py --config examples/pretraining_TSGP-2.yaml
```

Pretraining loads the pairs of the four archive folders of its corpus. It writes the
pairs it trained on to `trainingdata/train_pairs_<corpus>.pkl`, plus a test file of
further pairs, 10% of the training set's size. It keeps the checkpoint with the lowest
validation loss in `src/TSGP/transformer/models/`. A retrained model is a new sample, not
a copy of the published weights.
