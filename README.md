# Constrained Subgroup Discovery

This repository contains the code of two papers:

> Bach, Jakob. "Using Constraints to Discover Sparse and Alternative Subgroup Descriptions"

(To be published on arXiv.
Once it's published, we'll add a link to it here.
We'll link the experimental data, too.)

> Bach, Jakob, and Klemens Böhm. "..."

(To be published at a conference or in a journal.
Once it's published, we'll add a link to it here.
We'll link the experimental data, too.)

This document provides:

- An outline of the [repo structure](#repo-structure).
- Steps for [setting up](#setup) a virtual environment and [reproducing](#reproducing-the-experiments) the experiments.

## Repo Structure

Currently, the repository contains six Python files and four non-code files.
The non-code files are:

- `.gitignore`: For Python development.
- `LICENSE`: The software is MIT-licensed, so feel free to use the code.
- `README.md`: You are here :upside_down_face:
- `requirements.txt`: To set up an environment with all necessary dependencies; see below for details.

The code files comprise our experimental pipeline (see below for details):

- `prepare_datasets.py`: First stage of the experiments (download prediction datasets).
- `prepare_demo_datasets.py`: Alternative script for the first stage of the experiments,
  preparing fewer and smaller datasets (used in some preliminary benchmarking experiments).
- `run_experiments.py`: Second stage of the experiments (run subgroup discovery).
- `run_evaluation_(arxiv|short).py`: Third stage of the experiments (compute statistics and create plots for the paper).
- `data_handling.py`: Functions for working with prediction datasets and experimental data.

Additionally, we have organized the subgroup-discovery methods for our experiments as the standalone Python package `csd`,
located in the directory `csd_package/`.
See the corresponding [README](csd_package/README.md) for more information.

## Setup

Before running the scripts to reproduce the experiments, you should

1) Set up an environment (optional but recommended).
2) Install all necessary dependencies.

Our code is implemented in Python (version 3.8; other versions, including lower ones, might work as well).

### Option 1: `conda` Environment

If you use [`conda`](https://conda.io/), you can directly install the correct Python version into a new `conda` environment
and activate the environment as follows:

```bash
conda create --name <conda-env-name> python=3.8
conda activate <conda-env-name>
```

Choose `<conda-env-name>` as you like.

To leave the environment, run

```bash
conda deactivate
```

### Option 2: `virtualenv` Environment

We used [`virtualenv`](https://virtualenv.pypa.io/) (version 20.24.3; other versions might work as well)
to create an environment for our experiments.
First, you need to install the correct Python version yourself.
Let's assume the Python executable is located at `<path/to/python>`.
Next, you install `virtualenv` with

```bash
python -m pip install virtualenv==20.24.3
```

To set up an environment with `virtualenv`, run

```bash
python -m virtualenv -p <path/to/python> <path/to/env/destination>
```

Choose `<path/to/env/destination>` as you like.

Activate the environment in Linux with

```bash
source <path/to/env/destination>/bin/activate
```

Activate the environment in Windows (note the back-slashes) with

```cmd
<path\to\env\destination>\Scripts\activate
```

To leave the environment, run

```bash
deactivate
```

### Dependency Management

After activating the environment, you can use `python` and `pip` as usual.
To install all necessary dependencies for this repo, run

```bash
python -m pip install -r requirements.txt
```

If you make changes to the environment and you want to persist them, run

```bash
python -m pip freeze > requirements.txt
```

## Reproducing the Experiments

After setting up and activating an environment, you are ready to run the code.
Run

```bash
python -m prepare_datasets
```

to download and pre-process the input data for the experiments.

Next, start the experimental pipeline with

```bash
python -m run_experiments
```

Depending on your hardware, this might take some time.
For example, we had a runtime of roughly 34 hours on a server with an AMD EPYC 7551 CPU (32 physical cores, base clock of 2.0 GHz).

To print statistics and create the plots for the paper, run

```bash
python -m run_evaluation_arxiv
```

or

```bash
python -m run_evaluation_short
```

(The short version is more focused and therefore contains fewer evaluations.
Also, the plots are formatted a bit differently.)

All scripts have a few command-line options, which you can see by running the scripts like

```bash
python -m prepare_datasets --help
```
