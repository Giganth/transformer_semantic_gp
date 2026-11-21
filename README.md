# Transformer Semantic Genetic Programming

<p align="center">
  <a href="https://github.com/Giganth/transformer_semantic_gp/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/Giganth/transformer_semantic_gp/ci.yml?branch=main&amp;style=for-the-badge&amp;label=CI" alt="CI"></a>
  <a href="https://github.com/Giganth/transformer_semantic_gp/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-yellow?style=for-the-badge" alt="License"></a>
  <a href="https://arxiv.org/abs/2611.17455"><img src="https://img.shields.io/badge/arXiv-2611.17455-b31b1b?style=for-the-badge&amp;logo=arxiv&amp;logoColor=white" alt="arXiv"></a>
</p>

Transformer Semantic Genetic Programming (TSGP) is a semantic GP approach that uses a
pretrained transformer as its variation operator. The model is trained on pairs of
semantically similar programs and proposes, for a given parent, an offspring with similar
semantics. Unlike semantic operators built on fixed syntactic transformations, TSGP
learns diverse structural variations.

## Contents

- [Versions and papers](#versions-and-papers)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Datasets](#datasets)
- [Running experiments](#running-experiments)
- [Building your own model](#building-your-own-model)
- [Repository structure](#repository-structure)
- [Citation](#citation)
- [License](#license)

## Versions and papers

| Version | Paper | Novelty | Model |
|---|---|---|---|
| [2](https://github.com/Giganth/transformer_semantic_gp/tree/TSGP2) | Anthes, Geiger, Briesch, and Rothlauf, 2026, [arXiv:2611.17455](https://arxiv.org/abs/2611.17455) — *TSGP-2: On the Non-Linearity of Symbolic Regression Problems and How to Exploit It with Model-based Semantic Genetic Programming* | Improved training data | `TSGP-2.weights.h5` and `TSGP.weights.h5` in this repository |
| [1](https://github.com/Giganth/transformer_semantic_gp/tree/TSGP) | Anthes, Sobania, and Rothlauf, 2026, [DOI 10.1145/3845808](https://dl.acm.org/doi/10.1145/3845808) — *Transformer Semantic Genetic Programming for d-dimensional Symbolic Regression Problems* (TELO 2026) | Multi-dimensional model | `TSGP.weights.h5` in tag `TSGP`; the original model of the paper runs on [figshare](https://doi.org/10.6084/m9.figshare.30601235) (activate `legacy_mode`) |
| [1](https://github.com/Giganth/transformer_semantic_gp/tree/TSGP) | Anthes, 2026, [DOI 10.1007/978-3-032-23005-8_4](https://doi.org/10.1007/978-3-032-23005-8_4) — *Syntactic Flexibility Enables Compact Solutions in Transformer Semantic GP* (EuroGP 2026, pp. 51–66) | Syntactical variation analysis | `TSGP.weights.h5` in tag `TSGP`; the original model of the paper runs on [figshare](https://doi.org/10.6084/m9.figshare.30601235) (activate `legacy_mode`) |
| – | Anthes, Sobania, and Rothlauf, 2025, [DOI 10.1145/3712256.3726412](https://dl.acm.org/doi/10.1145/3712256.3726412) — *Transformer Semantic Genetic Programming for Symbolic Regression* (GECCO 2025) | Learning semantic variations | not supported in this implementation |

## Installation

Tested with Python 3.12 on Linux with an NVIDIA GPU (CUDA 12, installed through pip).

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
```

## Quick start

One search with each operator and, for comparison, with standard GP (problem, seed and all
settings are in the config files):

```bash
python examples/symbolic_regression.py --config examples/transformer_gp_TSGP.yaml     # TSGP
python examples/symbolic_regression.py --config examples/transformer_gp_TSGP-2.yaml   # TSGP-2
python examples/symbolic_regression.py --config examples/standard_gp.yaml             # Standard GP
```

`--dataset feynman_II_13_17` runs the same configuration on another problem.

## Datasets

`src/TSGP/symbolic_regression/datasets/` contains six example problems from PMLB (three
Feynman, three real-world; features in the leading columns, target in the last one).
Further problems are added by placing a CSV in the same format there and selecting it with
`problem.dataset`. Datasets with more than 10,000 rows are subsampled
(`problem.subsample: True`). `Semantic_Testset.csv` is a randomly generated sample of 500
standard-normal input points on which the semantics of a program can be measured (necessary
for training).

## Running experiments

📖 **[Experiments Guide](examples/EXPERIMENTS_GUIDE.md)** - Detailed instructions for:
- Running single experiments for quick testing
- Setting up batch experiments with multiple runs
- Parallel execution on clusters using taskspooler
- Analyzing and visualizing results

30 runs (seeds 1000–1029) per problem and method:

```bash
python src/TSGP/experiments/initialize.py \
    -f experiments_data/TSGP-2_feynman_I_34_27 \
    -a transformer_gp \
    -c examples/transformer_gp_TSGP-2.yaml \
    -d feynman_I_34_27 -n 30 -r 1000
python src/TSGP/experiments/run.py -f experiments_data/TSGP-2_feynman_I_34_27 -n 30
python src/TSGP/experiments/analyze.py -f experiments_data/TSGP-2_feynman_I_34_27 -o results.pdf
```

Every run is seeded (`random_seed`) and reproduces bit for bit on the same hardware and
software stack. The transformer samples tokens from probabilities computed on the GPU,
though, and different GPUs or CUDA/TensorFlow versions round them differently. A single
different token changes all later random draws, so runs on other hardware diverge from
published ones within the first generations; the results over many seeds do not change.

## Building your own model

The repository includes the pretrained weights. To train your own model, follow the
comprehensive guide:

<p align="center">
  <img src="misc/model_building.png" alt="Model building" width="90%">
</p>

📖 **[Model Building Guide](examples/MODEL_BUILDING_GUIDE.md)** - Complete walkthrough of:
1. **Data Generation** - Generate training data using evolutionary search
2. **Semantic Similarity Search** - Build similarity mappings between expressions
3. **Pretraining** - Train the Transformer model on the generated data

## Repository structure

```
src/TSGP/
├── algorithms/           # Standard GP, Transformer GP and the archive data generator
├── core/                 # evolutionary engine, primitive set, monitoring
├── transformer/          # model, sampling, pretraining, similarity search, trained models
├── symbolic_regression/  # datasets, evaluation, synthetic target functions
├── experiments/          # batch runs: initialize.py, run.py, analyze.py
└── config/               # defaults and YAML loader
examples/                 # configs of both models and entry points (search, data generation, pretraining)
misc/                     # figure used in the README
```

## Citation

If you use TSGP in your research, cite the paper of the version you use (see the table
above). The paper of this version is:

```bibtex
@misc{anthes2026tsgp2,
  title={TSGP-2: On the Non-Linearity of Symbolic Regression Problems and How to Exploit It with Model-based Semantic Genetic Programming},
  author={Anthes, Philipp and Geiger, Alina and Briesch, Martin and Rothlauf, Franz},
  year={2026},
  eprint={2611.17455},
  archivePrefix={arXiv},
  primaryClass={cs.NE},
  doi={10.48550/arXiv.2611.17455}
}
```

TSGP is built on [DEAP](https://github.com/DEAP/deap):

```bibtex
@article{DEAP_JMLR2012,
  author  = {Fortin, F\'elix-Antoine and De Rainville, Fran\c{c}ois-Michel and
             Gardner, Marc-Andr\'e and Parizeau, Marc and Gagn\'e, Christian},
  title   = {{DEAP}: Evolutionary Algorithms Made Easy},
  journal = {Journal of Machine Learning Research},
  volume  = {13},
  pages   = {2171--2175},
  year    = {2012}
}
```

## License

MIT, see [LICENSE](LICENSE).
