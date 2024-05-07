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
- A short [overview and demo](#functionality-and-demo) of the core (subgroup-discovery) functionality.
- [Guidelines for developers](#developer-info) who want to modify or extend the code base.
- Steps for [setting up](#setup) a virtual environment and [reproducing](#reproducing-the-experiments) the experiments.

## Repo Structure

Currently, the repository contains seven Python files and four non-code files.
The non-code files are:

- `.gitignore`: For Python development.
- `LICENSE`: The software is MIT-licensed, so feel free to use the code.
- `README.md`: You are here :upside_down_face:
- `requirements.txt`: To set up an environment with all necessary dependencies; see below for details.

Six of the code files comprise our experimental pipeline (see below for details):

- `prepare_datasets.py`: First stage of the experiments (download prediction datasets).
- `prepare_demo_datasets.py`: Alternative script for the first stage of the experiments,
  preparing fewer and smaller datasets (used in some preliminary benchmarking experiments).
- `run_experiments.py`: Second stage of the experiments (run subgroup discovery).
- `run_evaluation_(arxiv|short).py`: Third stage of the experiments (compute statistics and create plots for the paper).
- `data_handling.py`: Functions for working with prediction datasets and experimental data.

In contrast, `sd.py` files contain classes and functions for subgroup discovery and may also be used as a standalone module.

## Functionality and Demo

Currently, we provide seven subgroup-discovery methods as classes in `sd.py`:

- solver-based search: `MIPSubgroupDiscoverer`, `SMTSubgroupDiscoverer`
- heuristic search: `BeamSearchSubgroupDiscoverer`, `BestIntervalSubgroupDiscoverer`, `PRIMSubgroupDiscoverer`
- baselines (simple heuristics): `MORSSubgroupDiscoverer`, `RandomSubgroupDiscoverer`

The heuristics are from literature or adaptations from other packages,
while we conceived the solver-based methods and baselines.
All subgroup-discovery methods can discover subgroups (who would have guessed?),
while some of them can also find alternative subgroup descriptions.

Further, we provide four evaluation metrics for binary (`bool` or `int`) subgroup-membership vectors of data objects:

- `wracc()`: Weighted Relative Accuracy
- `nwracc()`: Weighted Relative Accuracy normalized to `[-1, 1]`
- `jaccard()`: Jaccary similarity (1 - Jaccard distance)
- `hamming()`: Normalized Hamming similarity (1 - Hamming distance normalized to `[0, 1]`; equals prediction accuracy)

For each of them, there is a general version (accepting `pd.Series`, `np.array`, or even simpler sequences like plain lists)
and a faster, `numpy`-specfic version with the same functionality (suffixed `_np`).

### Discovering (Original) Subgroups

Using a subgroup-discovery method from `sd.py` is similar to working with prediction models from `scikit-learn`.
All subgroup-discovery methods are implemented as subclasses of `SubgroupDiscoverer` and provide the following functionality:

- `fit(X, y)` and `predict(X)`, as in `scikit-learn`, working with `pandas.DataFrame` and `pandas.Series`
- `evaluate(X_train, y_train, X_test, y_test)`, which combines fitting and prediction on a train-test split of a dataset
- `get_box_lbs()` and `get_box_ubs()`, which return the lower and upper bounds on features in the subgroup description (as `pandas.Series`)
- `is_feature_selected()` and `get_selected_feature_idxs()`, which provide information on selected (= restricted, used) features in the subgroup description (as lists)

The actual subgroup discovery occurs during fitting.
The passed dataset has to be purely numeric (categorical columns should be encoded accordingly) with a binary target,
with the class label `1` being the class of interest and `0` being the other class.
Any parameters for the search have to be passed before fitting,
i.e., already when initializing the instance of the subgroup-discovery method.
While some parameters are specific to the subgroup-discovery method,
all existing methods have a parameter `k` to limit (upper bound) the number of features selected in the subgroup description.
The prediction routine classifies data objects as belonging to the subgroup (= 1) or not (= 0).
`sd.py` also contains functions to evaluate predictions,
though metrics for binary classification from `sklearn.metrics` should work as well.

Here is a small example for initialization, fitting, prediction, and evaluation:

```python
import sd
import sklearn.datasets
import sklearn.model_selection

X, y = sklearn.datasets.load_iris(as_frame=True, return_X_y=True)
y = (y == 1).astype(int)  # binary classification (in original data, three classes)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, train_size=0.8, random_state=25)

model = sd.BeamSearchSubgroupDiscoverer(beam_width=10)
fitting_result = model.fit(X=X_train, y=y_train)
print('Fitting result:', fitting_result)
print('Train WRAcc:', round(sd.wracc(y_true=y_train, y_pred=model.predict(X=X_train)), 2))
print('Test WRAcc:', round(sd.wracc(y_true=y_test, y_pred=model.predict(X=X_test)), 2))
print('Lower bounds:', model.get_box_lbs().tolist())
print('Upper bounds:', model.get_box_ubs().tolist())
```

The output of this code snippet looks like this (optimization time may vary):

```
Fitting result: {'objective_value': 0.208125, 'optimization_status': None, 'optimization_time': 0.03125}
Train WRAcc: 0.21
Test WRAcc: 0.21
Lower bounds: [-inf, -inf, 3.0, -inf]
Upper bounds: [7.0, inf, 5.1, 1.7]
```

Beam search optimizes Weighted Relative Accuracy (WRAcc), so objective value and train WRAcc (without rounding) are equal.
The `optimization_status` only is relevant for solver-based subgroup-discovery methods.
If a feature is not restricted regarding lower or upper bound, the corresponding bounds are infinite.

`evaluate()` combines fitting, prediction, and evaluation:

```python
import sd
import sklearn.datasets
import sklearn.model_selection

X, y = sklearn.datasets.load_iris(as_frame=True, return_X_y=True)
y = (y == 1).astype(int)  # binary classification (in original data, three classes)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, train_size=0.8, random_state=25)

model = sd.BeamSearchSubgroupDiscoverer(beam_width=10)
evaluation_result = model.evaluate(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
print(evaluation_result.transpose())
```

The output is a `pandas.DataFrame` (optimization time and fitting time may vary):

```
                                             0
objective_value                       0.208125
optimization_status                       None
optimization_time                      0.03125
fitting_time                          0.046875
train_wracc                           0.208125
test_wracc                            0.212222
train_nwracc                          0.975904
test_nwracc                           0.864253
box_lbs                [-inf, -inf, 3.0, -inf]
box_ubs                   [7.0, inf, 5.1, 1.7]
selected_feature_idxs                [0, 2, 3]
```

Fitting time may be higher than optimization time since it also includes preparing optimization and the results object.
However, except for solver-based optimization (where entering all constraints into the solver may take some time),
the difference between optimization time and fitting time should be marginal.
The evaluation routine computes WRAcc, whose minimum and maximum depend on the class imbalance in the dataset,
as well as nWRAcc, which is normalized to `[-1, 1]`.

### Discovering Alternative Subgroup Descriptions

Additionally, subgroup-discovery methods inheriting from `AlternativeSubgroupDiscoverer`
(currently only `BeamSearchSubgroupDiscoverer` and `SMTSubgroupDiscoverer`)
can search alternative subgroup descriptions with the method `search_alternative_descriptions()`.
Before starting this search, you should set `a` (number of alternatives) and
`tau_abs` (number of previously selected features that must not be selected again) in the initializer.
Also, we highly recommend limiting the number of selected features with the parameter `k` in the initializer,
so there are still enough unselected features that may be selected in the alternatives instead.

```python
import sd
import sklearn.datasets
import sklearn.model_selection

X, y = sklearn.datasets.load_iris(as_frame=True, return_X_y=True)
y = (y == 1).astype(int)  # binary classification (in original data, three classes)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, train_size=0.8, random_state=25)

model = sd.BeamSearchSubgroupDiscoverer(beam_width=10, k=2, tau_abs=1, a=4)
search_results = model.search_alternative_descriptions(X_train=X_train, y_train=y_train,
                                                       X_test=X_test, y_test=y_test)
print(search_results.drop(columns=['box_lbs', 'box_ubs', 'selected_feature_idxs']).round(2).transpose())
```

This code snippet outputs a `pandas.DataFrame` (optimization time and fitting time may vary):

```
objective_value      0.21  0.94  0.92  0.81  0.76
optimization_status  None  None  None  None  None
optimization_time    0.02  0.03  0.02  0.03  0.02
fitting_time         0.03  0.03  0.02  0.03  0.02
train_wracc          0.21  0.17  0.16   0.1   0.1
test_wracc           0.21  0.17  0.17  0.13  0.11
train_nwracc         0.98  0.82  0.77  0.46  0.45
test_nwracc          0.86  0.71  0.69  0.54  0.46
alt.hamming           1.0  0.94  0.92  0.81  0.76
alt.jaccard           1.0  0.83  0.77   0.5  0.48
alt.number              0     1     2     3     4
```

The first subgroup (with an `alt.number` of `0`) is the same as if using the conventional `fit()` routine.
The objective value corresponds to train WRAcc for this subgroup and normalized Hamming similarity for the subsequent subgroups.
You can see that the similarity of alternative subgroup descriptions to the original subgroup
(`alt.hamming` and `alt.jaccard`) decreases over the number of alternatives (`alt.number`),
as does the WRAcc on the training set and test set.

## Developer Info

If you want to add another subgroup-discovery method, make it a subclass of `SubgroupDiscoverer`.
As a minimum, you should

- add the initializer `__init__()` to set parameters of your subgroup-discovery method
  (also call `super().__init__()` for the initialization from the superclass) and
- override the `fit()` method to search for subgroups.

`fit()` needs to set the fields `_box_lbs` and `_box_ubs` (inherited from the superclass)
to contain the features' lower and upper bounds in the subgroup description (as `pandas.Series`).
If `_box_lbs` and `_box_ubs` are set as fields, `predict()` works automatically with these bounds and need not be overridden.
For integration into the experimental pipeline, `fit()` should also return a dictionary
with the keys `objective_value`, `optimization_status` (may be `None`), and `optimization_time`.
If your subgroup-discovery method is not tailored towards a specific objective,
you may use the top-level functions `wracc()` or `wracc_np()` from `sd.py` to guide the search
(the former is slower but supports more data types, the latter is tailored to `numpy` arrays and faster).

To support feature-cardinality constraints, like the other subgroup-discovery methods,
you may want to allow setting an upper bound on the number of selected features `k`
during initialization and observing it during fitting.
We propose to store `k` in a field called `_k`;
however, it is not used in the methods of superclass `SubgroupDiscoverer`.

If your subgroup-discovery method should also support the search for alternative subgroup descriptions,
make it a subclass of `AbstractSubgroupDiscoverer` (which inherits from `SubgroupDiscoverer`).
The search for alternatives and the fitting routine are already implemented there.
In particular, `fit()` dispatches to the method `_optimize()`, which you need to override.
The latter should both be able to

- search for the original subgroup, optimizing WRAcc or another notion of subgroup quality and
- search for alternative subgroup descriptions by
  - optimizing normalized Hamming similarity (see top-level functions `hamming()` and `hamming_np` in `sd.py`)
    to the original subgroup (or another notion of subgroup similarity) and
  - having constraint that at least `tau_abs` features from each previous subgroup description need to be de-selected
    (though not more features than actually were selected, to prevent infeasibilities).

Your implementation of `_optimize()` has to automatically switch the objective and add the constraints for alternatives if necessary.
The method's parameters `was_feature_selected_list` and `was_instance_in_box` tell you, if not `None`,
that you should search for an alternative subgroup description; else, search for the original subgroup.
Additionally, you should add parameters `a` and `tau_abs` in the initializer of your subgroup-discovery class,
storing them in the fields `_a` and `_tau_abs`.
The latter two fields are automatically used in `evaluate()` (to decide whether to search only for an original subgroup
or also for alternative subgroup descriptions) and should be used in your implementation of `_optimize()`.

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
