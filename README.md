# Transformer Semantic Genetic Programming

<p align="center">
  <a href="https://github.com/Giganth/transformer_semantic_gp/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/Giganth/transformer_semantic_gp/ci.yml?branch=main&amp;style=for-the-badge&amp;label=CI" alt="CI"></a>
  <a href="https://github.com/Giganth/transformer_semantic_gp/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-yellow?style=for-the-badge" alt="License"></a>
  <a href="https://dl.acm.org/doi/10.1145/3845808"><img src="https://img.shields.io/badge/TELO-10.1145%2F3845808-blue?style=for-the-badge&amp;logo=acm&amp;logoColor=white" alt="TELO 2026"></a>
</p>

Transformer Semantic Genetic Programming (TSGP) is a semantic GP approach that uses a pre-trained
transformer model as a variation operator to generate offspring programs with high semantic similarity
to a given parent. Unlike other semantic GP approaches that rely on fixed syntactic transformations, TSGP
learns diverse structural variations that lead to offspring with similar semantics.

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
| [2](https://github.com/Giganth/transformer_semantic_gp/tree/TSGP2) | Anthes, Geiger, Briesch, and Rothlauf, 2026, [arXiv:2611.17455](https://arxiv.org/abs/2611.17455) — *TSGP-2: On the Non-Linearity of Symbolic Regression Problems and How to Exploit It with Model-based Semantic Genetic Programming* | Improved training data | `TSGP-2.weights.h5` in tag `TSGP2` |
| [1](https://github.com/Giganth/transformer_semantic_gp/tree/TSGP) | Anthes, Sobania, and Rothlauf, 2026, [DOI 10.1145/3845808](https://dl.acm.org/doi/10.1145/3845808) — *Transformer Semantic Genetic Programming for d-dimensional Symbolic Regression Problems* (TELO 2026) | Multi-dimensional model | `TSGP.weights.h5` in this repository; the original model of the paper runs on [figshare](https://doi.org/10.6084/m9.figshare.30601235) (activate `legacy_mode`) |
| [1](https://github.com/Giganth/transformer_semantic_gp/tree/TSGP) | Anthes, 2026, [DOI 10.1007/978-3-032-23005-8_4](https://doi.org/10.1007/978-3-032-23005-8_4) — *Syntactic Flexibility Enables Compact Solutions in Transformer Semantic GP* (EuroGP 2026, pp. 51–66) | Syntactical variation analysis | `TSGP.weights.h5` in this repository; the original model of the paper runs on [figshare](https://doi.org/10.6084/m9.figshare.30601235) (activate `legacy_mode`) |
| – | Anthes, Sobania, and Rothlauf, 2025, [DOI 10.1145/3712256.3726412](https://dl.acm.org/doi/10.1145/3712256.3726412) — *Transformer Semantic Genetic Programming for Symbolic Regression* (GECCO 2025) | Learning semantic variations | not supported in this implementation |

## Installation
```bash
# Create a virtual environment
python3.12 -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Optional: upgrade pip
python -m pip install --upgrade pip

# Install the package
pip install -e .
```

## Quick Start

**Standard GP:**
```bash
./venv/bin/python examples/symbolic_regression.py --algorithm standard_gp
```

**Transformer GP:**
```bash
./venv/bin/python examples/symbolic_regression.py --algorithm transformer_gp
```
## Key Features

- **Semantic-aware variation** using a transformer model
- **FAISS-based similarity search** for nearest neighbor selection
- **Pre-trained weights** for TSGP (`src/TSGP/transformer/models/TSGP.weights.h5`).
- **Configurable experiments** via YAML files

## Project Structure

```
src/TSGP/
├── algorithms/          # Variation Operators of Standard GP and Transformer GP 
├── transformer/         # Transformer model, pretraining, and similarity search
├── core/                # Evolutionary search (engine.py) & core functionalities
├── symbolic_regression/ # Dataset loading, evaluation, and benchmark datasets
└── experiments/         # Batch experiment management (initialize, run, analyze)

examples/              # Example scripts and configs
experiments_data/      # Generated archives, logs, and experiment outputs

```

## Requirements

- Python ≥3.10, ≤3.13 (tested on 3.10)
- TensorFlow ≥2.8 (with CUDA support)
- FAISS, DEAP, NumPy, Pandas, scikit-learn

See [`pyproject.toml`](pyproject.toml) or [`requirements.txt`](requirements.txt) for complete dependencies with exact versions.

## Running Experiments

📖 **[Experiments Guide](examples/EXPERIMENTS_GUIDE.md)** - Detailed instructions for:
- Running single experiments for quick testing
- Setting up batch experiments with multiple runs
- Parallel execution on clusters using taskspooler
- Analyzing and visualizing results

## Model Building
The repository includes pre-trained transformer weights (`TSGP.weights.h5`). The training data used to train the transformer is available on [Figshare](https://doi.org/10.6084/m9.figshare.30601235). To train your own model, follow the comprehensive guide:

<p align="center">
  <a href="misc/TSGP_Model_Building.pdf">
    <img src="misc/model_building.png" alt="Model Building Guide" width="90%">
  </a>
</p>

📖 **[Model Building Guide](examples/MODEL_BUILDING_GUIDE.md)** - Complete walkthrough of:
1. **Data Generation** - Generate training data using evolutionary search
2. **Semantic Similarity Search** - Build similarity mappings between expressions
3. **Pretraining** - Train the Transformer model on the generated data

## Configuration

Edit YAML files in `examples/` to customize:
- Algorithm settings: Population size, generations, selection methods
- Variation operators: Crossover/mutation rates, transformer semantic variation (e.g. target semantic distance)
- Transformer architecture: Layers, attention heads, embedding dimensions
- Dataset selection: Choose from built-in benchmarks or add custom datasets


## Citation
If you use this code, cite:

```bibtex
@article{anthes2026dimensional,
  title={Transformer Semantic Genetic Programming for d-dimensional Symbolic Regression Problems},
  author={Anthes, Philipp and Sobania, Dominik and Rothlauf, Franz},
  journal={ACM Transactions on Evolutionary Learning and Optimization},
  volume={1},
  number={1},
  year={2026},
  doi={10.1145/3845808}
}
```

The conference version is *Transformer Semantic Genetic Programming for Symbolic Regression*
(GECCO 2025, [DOI 10.1145/3712256.3726412](https://dl.acm.org/doi/10.1145/3712256.3726412)).

> **Note:** This repository is based on the paper *Transformer Semantic Genetic Programming for d-dimensional Symbolic Regression Problems* (Anthes, Sobania, and Rothlauf, 2026, [DOI 10.1145/3845808](https://dl.acm.org/doi/10.1145/3845808)). The codebase has been completely refactored and a new improved transformer model is trained, so results may diverge slightly from those reported in the paper. If you want to use the model used for the experiments, activate `legacy_mode` and download the model here: https://doi.org/10.6084/m9.figshare.30601235.

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