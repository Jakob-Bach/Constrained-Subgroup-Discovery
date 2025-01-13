# Constrained Subgroup Discovery

This repository contains the code for

- two papers,
- (parts of) a dissertation,
- and the Python package [`csd`](https://pypi.org/project/csd/).

This document provides:

- An overview of the [related publications](#publications).
- An outline of the [repo structure](#repo-structure).
- Steps for [setting up](#setup) a virtual environment and [reproducing](#reproducing-the-experiments) the experiments.

## Publications

> Bach, Jakob. "Using Constraints to Discover Sparse and Alternative Subgroup Descriptions"

is published on [arXiv](https://arxiv.org/).
You can find the paper [here](https://doi.org/10.48550/arXiv.2406.01411).
You can find the corresponding complete experimental data (inputs as well as results) on [*RADAR4KIT*](https://doi.org/10.35097/caKKJCtoKqgxyvqG).
Use the tags `run-2024-05-13` and `evaluation-2024-05-15` for reproducing the experiments.

> Bach, Jakob. "..."

(To be published at a conference or in a journal.
Once it's published, we'll add a link to it here.
We'll link the experimental data, too.)

> Bach, Jakob. "Leveraging Constraints for User-Centric Feature Selection"

is a dissertation in progress.
Once it is published, we will link it here as well.
You can find the corresponding complete experimental data (inputs as well as results) on [*RADAR4KIT*](https://doi.org/10.35097/4kjyeg0z2bxmr6eh).
Use the tags `run-2024-05-13-dissertation` and `evaluation-2024-11-02-dissertation` for reproducing the experiments.

## Repo Structure

Currently, the repository consists of three directories and three files at the top level.

The top-level files are:

- `.gitignore`: For Python development.
- `LICENSE`: The software is MIT-licensed, so feel free to use the code.
- `README.md`: You are here :upside_down_face:

The directory `csd_package/` contains the Python package `csd`,
which comprises the subgroup-discovery methods and evaluation metrics we implemented for our experiments.
You can use this package independently of our experiments.
See the corresponding [README](csd_package/README.md) for more information.

The directory `main_experiments/` contains the experiments described in the publications listed above.

- `prepare_datasets.py`: First stage of the experiments (download prediction datasets).
- `run_experiments.py`: Second stage of the experiments (run subgroup discovery).
- `run_evaluation_(arxiv|dissertation|short).py`: Third stage of the experiments (compute statistics and create plots).
- `data_handling.py`: Functions for working with prediction datasets and experimental data.
- `sd4py_methods.py`: Classes wrapping methods from the package `sd4py` for our main experimental pipeline
  (the remaining subgroup-discovery methods are implemented by us and reside in the package `csd`).
- `requirements.txt`: To set up an environment with all necessary dependencies; see below for details.

The directory `competitor_runtime_experiments/` contains additional experiments,
which are not described in the publications but helped us choose competitors for our main experiments.
In particular, we benchmarked the runtime of subgroup-discovery methods from the packages
`pysubdisc`, `pysubgroup`, `sd4py`, and `subgroups` (besides our package `csd`).

- `prepare_competitor_runtime_datasets.py`: First stage of the experiments (download prediction datasets).
- `run_competitor_runtime_experiments.py`: Second stage of the experiments (run subgroup discovery).
- `run_competitor_runtime_evaluation`: Third stage of the experiments (compute statistics).
- `data_handling.py`: Functions for working with prediction datasets and experimental data (copied from main experiments).
- `runtime_competitors.py`: Classes wrapping subgroup-discovery methods from multiple packages for our competitor-runtime pipeline.
- `requirements.txt`: To set up an environment with all necessary dependencies (extends requirements of main experiments).

## Setup

Before running the scripts to reproduce the experiments, you should

1) Install Python (`3.8` for main experiments, `3.9` for competitor-runtime experiments)
   and make sure the command `python` works from the command line (check `python --version`).
2) Install Java 8 and make sure the command `java` works from the command line (check `java -version`).
   While our implementation of subgroup-discovery methods (in the package `csd`) does not require Java,
   the packages `pysubdisc` and `sd4py` (used in `run_experiments.py` and `run_competitor_runtime_experiments.py`) do.
3) Set up an environment for Python (optional but recommended).
4) Install all necessary Python dependencies.

In the following, we describe steps (3) and (4) in detail.

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
To install all necessary dependencies, go into the directory `main_experiments/` or `competitor_runtime_experiments/`
(their requirements differ; the latter uses a superset of the former) and run

```bash
python -m pip install -r requirements.txt
```

If you make changes to the environment and you want to persist them, run

```bash
python -m pip freeze > requirements.txt
```

## Reproducing the Experiments

After setting up and activating an environment, you are ready to run the code.
You need to run three Python scripts, which differ between [main experiments](#main-experiments)
and [competitor-runtime experiments](#competitor-runtime-experiments).
All scripts have a few command-line options, which you can see by running the scripts with the `--help` flag like

```bash
python -m prepare_datasets --help
```

### Main Experiments

First, download and pre-process the input data for the experiments with

```bash
python -m prepare_datasets
```

Next, start the experimental pipeline with

```bash
python -m run_experiments
```

Depending on your hardware, this might take some time.
For example, we had a runtime of roughly 34 hours on a server with an AMD EPYC 7551 CPU (32 physical cores, base clock of 2.0 GHz).

Finally, print statistics and create the plots with

```bash
python -m run_evaluation_<<version>>
```

`<<version>>` can be `arxiv`, `dissertation`, or `short`.
The evaluation length differs between versions, as does the plot formatting.
The arXiv version has the longest and most detailed evaluation.

### Competitor-Runtime Experiments

First, download and pre-process the input data for the experiments with

```bash
python -m prepare_competitor_runtime_datasets
```

Next, start the experimental pipeline with

```bash
python -m run_competitor_runtime_experiments
```

Unfortunately, the packages `pysubdisc` and `sd4py` both start a Java Virtual Machine but with different dependencies,
which causes tasks to crash.
`sd4py` even starts the JVM just when loading the package, while `pysubdisc` starts it when needed.
Thus, you should run the pipeline twice:

1) With `sd4py` methods but not `pysubdisc` methods.
  To this end, remove (or comment) the latter from the dict `SD_METHODS` in `run_competitor_runtime_experiments.py`.
2) With `pysubdisc` methods but not `sd4py` methods.
  To this end, remove (or comment) the latter from the dict `SD_METHODS` in `run_competitor_runtime_experiments.py`.
  Additional, remove (or comment) the `sd4py` import in `runtime_competitor.py`
  and all classes that depend on it (`SD4PyMethod` and three subclasses).

The remaining subgroup-discovery methods can be included in either run.

Finally, print statistics with

```bash
python -m run_competitor_runtime_evaluation
```
