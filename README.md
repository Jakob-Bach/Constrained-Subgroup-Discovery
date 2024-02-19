# Constrained Subgroup Discovery

This repository contains the code of the paper

> Bach, Jakob. "Constrained Subgroup Discovery"

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

Currently, we provide seven subgroup-discovery methods as classes in `sd.py`:

- exact optimization: `MIPSubgroupDiscoverer`, `SMTSubgroupDiscoverer`
- heuristics: `BeamSearchSubgroupDiscoverer`, `BestIntervalSubgroupDiscoverer`, `PRIMSubgroupDiscoverer`
- baselines: `MORBSubgroupDiscoverer`, `RandomSubgroupDiscoverer`

The heuristics are from literature or adaptations from other packages, while we conceived the remaining methods.

Using a subgroup-discovery method from `sd.py` is similar to working with prediction models from `scikit-learn`.
All subgroup-discovery methods are implemented as subclasses of `SubgroupDiscoverer` and provide the following functionality:

- `fit(X, y)` and `predict(X)`, as in `scikit-learn`, working with `pandas.DataFrame` and `pandas.Series`
- `evaluate(X_train, y_train, X_test, y_test)`, which combines fitting and prediction on a train-test split of a dataset
- `get_box_lbs()` and `get_box_ubs()`, which return the lower and upper bounds on features for the subgroup (as `pandas.Series`)
- `is_feature_selected()` and `get_selected_feature_idxs()`, which provide informationon on selected (restricted) features in the subgroup (as lists)

The actual subgroup discovery occurs during fitting.
The passed dataset has to be purely numeric (categorical columns should be encoded) with a binary target,
with class label `1` being the class of interest and `0` being the other class.
Any parameters for the search have to be passed beforehand,
i.e., when initializing the instance of the subgroup-discovery method.
While some parameters are specific to the subgroup-discovery method,
all existing methods have a parameter `k` to limit (upper bound) the number of features selected (restricted) in the subgroup.
The prediction routine classifies data objects as belonging to the subgroup (= 1) or not (= 0).
`sd.py` also contains functions to evaluate predictions (`wracc()`, `jaccard()`, and `hamming()`),
but metrics for binary classification from `sklearn.metrics` should work as well.

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
The `optimization_status` only plays a role for exact (optimizer-based) subgroup-discovery methods.
If a feature is not restricted regarding lower or upper bound, the corresponding bounds are infinite.

`evaluate()` combines fitting, prediction, and evaluation:

```
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
box_lbs                [-inf, -inf, 3.0, -inf]
box_ubs                   [7.0, inf, 5.1, 1.7]
selected_feature_idxs                [0, 2, 3]
```

Fitting time may be higher than optimization time since it also includes preparing optimization and the results object.
However, except for exact optimization (where entering all constraints into the optimizer may take some time),
the difference between optimization time and fitting time should be marginal.

Additionally, subgroup-discovery methods inheriting from `AlternativeSubgroupDiscoverer`
(currenty only `BeamSearchSubgroupDiscoverer` and `SMTSubgroupDiscoverer`)
can search alternative subgroup descriptions with the method `search_alternative_descriptions()`.
Before starting this search, you should set `a` (number of alternatives) and
`tau_abs` (number of previously selected features that must not be selected again) in the initializer.
Also, we highly recommend to limit the number of selected features with the parameter `k` in the initializer,
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
alt.hamming           1.0  0.94  0.92  0.81  0.76
alt.jaccard           1.0  0.83  0.77   0.5  0.48
alt.number              0     1     2     3     4
```

The first subgroup (with an `alt.number` of `0`) is the same as if using the conventional `fit()` routine.
The objective value corresponds to train WRAcc for this subgroup and Hamming similarity for the subsequent subgroups.
You can see that the similarity of alternative subgroup descriptions to the original subgroup
(`alt.hamming` and `alt.jaccard`) decreases over the number of alternatives (`alt.number`),
as does the WRAcc on the training set and test set.

## Developer Info

If you want to add another subgroup-discovery method, make it a subclass of `SubgroupDiscoverer`.
You need to override the `fit()` method such that it sets the fields `_box_lbs` and `_box_ubs`
to contain the lower and upper bounds on features in the subgroup (as `pandas.Series`).
If your subgroup-discovery method is not tailored towards a specific objective,
you may use the top-level functions `wracc()` or `wracc_np()` from `sd.py` to guide the search
(the former is slower but supports more data types, the latter is tailored to `numpy` arrays and faster).
If `_box_lbs` and `_box_ubs` are set, `predict()` works automatically with these bounds and need not be overridden.
To maintain full compatibility to all other subgroup-discovery methods,
you may want to allow setting an upper bound on the number of selected features `k`
during initialization and observing it during fitting.
We propose to store `k` in a field called `_k`;
however, it is not used in the methods of superclass `SubgroupDiscoverer`.
All other parameters for the subgroup-discovery method should also be set in the initializer and stored in fields.
Make sure to also call the initializer of the superclass when implementing the initializer of your class.

If your subgroup-discovery method should also support the search for alternative subgroup descriptions,
make it a subclass of `AbstractSubgroupDiscoverer` (which inherits from `SubgroupDiscoverer`).
The search for alternatives and the fitting routine are already implemented there.
In particular, `fit()` dispatches to the method `_optimize()`, which you need to override.
The latter should both be able to
- search for the original subgroup, optimizing WRAcc or another notion of subgroup quality
- search for alternative subgroup descriptions by
  - optimizing Hamming similarity (see top-level functions `hamming()` and `hamming_np` in `sd.py`)
    to original subgroup (or another notion of subgroup similarity),
  - having constraint that at least `tau_abs` features from each previous subgroup description need to be de-selected
    (though not more features than actually were selected, to prevent infeasibilities)
Your implementation of `_optimize()` has to switch the objective and add the constraints for alternatives.
The method's parameters `was_feature_selected_list` and `was_instance_in_box` tell you, if not `None`,
that you should search for an alternative subgroup description; else, search for the original subgroup.
Additionally, you should add parameters `a` and `tau_abs` in the initializer, storing them in the fields `_a` and `_tau_abs`.
The latter two fields are used in `evaluate()` (to decide whether to search only for an original subgroup
or also for alternative subgroup descriptions).

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
