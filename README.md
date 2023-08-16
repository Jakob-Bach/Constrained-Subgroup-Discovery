# Optimal Subgroup Discovery

This repository contains the code of the paper

> Bach, Jakob. "Optimal Subgroup Discovery"

(The paper is not published yet.
Once it's published, we'll add a link to it here.
We'll link the experimental data, too.)

This document provides:

- An outline of the repo structure.
- A short demo of the core functionality.
- Guidelines for developers who want to modify or extend the code base.
- Steps to reproduce the experiments, including setting up a virtual environment.

## Repo Structure

Currently, the repository contains five Python files and four non-code files.
The non-code files are:

- `.gitignore`: For Python development.
- `LICENSE`: The software is MIT-licensed, so feel free to use the code.
- `README.md`: You are here :upside_down_face:
- `requirements.txt`: To set up an environment with all necessary dependencies; see below for details.

Four of the code files are directly related to our experiments (see below for details):

- `prepare_datasets.py`: First stage of the experiments (download prediction datasets).
- `run_experiments.py`: Second stage of the experiments (run subgroup discovery).
- `run_evaluation.py`: Third stage of the experiments (compute statistics and create plots for the paper).
- `data_handling.py`: Functions for working with prediction datasets and experimental data.

`sd.py` files contain classes and functions for optimal subgroup discovery and may also be used as standalone module.

## Demo

## Developer Info

## Setup

Before running the scripts to reproduce the experiments, you should

1) Set up an environment (optional but recommended).
2) Install all necessary dependencies.

Our code is implemented in Python (version 3.8; other versions, including lower ones, might work as well).

### Option 1: `conda` Environment

If you use `conda`, you can directly install the correct Python version into a new `conda` environment
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

To print statistics and create the plots for the paper, run

```bash
python -m run_evaluation
```

All scripts have a few command-line options, which you can see by running the scripts like

```bash
python -m prepare_datasets --help
```
