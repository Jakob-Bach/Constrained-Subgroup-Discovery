# `csd` -- A Python Package for (Constrained) Subgroup Discovery

The package `csd` contains several subgroup-discovery methods and evaluation metrics.
Besides traditional (unconstrained) subgroup discovery,
we allow searching subgroup descriptions using a limited number of features (-> feature-cardinality constraint).
Additionally, some methods allow searching alternative subgroup descriptions,
which should replicate a given subgroup membership as good as possible while using different features in the description.

This document provides:

- Steps for [setting up](#setup) the package.
- A short [overview](#functionality) of the (subgroup-discovery) functionality.
- A [demo](#demo) for discovering subgroups and alternative subgroup descriptions.
- [Guidelines for developers](#developer-info) who want to modify or extend the code base.

If you use this package for a scientific publication, please cite [our paper](https://doi.org/10.48550/arXiv.2406.01411)

```
@misc{bach2024using,
	title={Using Constraints to Discover Sparse and Alternative Subgroup Descriptions},
	author={Bach, Jakob},
	howpublished={arXiv:2406.01411v1 [cs.LG]},
	year={2024},
	doi={10.48550/arXiv.2406.01411},
}
```

## Setup

You can install our package from [PyPI](https://pypi.org/):

```
python -m pip install csd
```

Alternatively, you can install the package from GitHub:

```bash
python -m pip install git+https://github.com/Jakob-Bach/Constrained-Subgroup-Discovery.git#subdirectory=csd_package
```

If you already have the source code for the package (i.e., the directory in which this `README` resides)
as a local directory on your computer (e.g., after cloning the project), you can also perform a local install:

```bash
python -m pip install csd_package/
```

## Functionality

Currently, we provide seven subgroup-discovery methods as classes:

- solver-based search: `MIPSubgroupDiscoverer`, `SMTSubgroupDiscoverer`
- heuristic search: `BeamSearchSubgroupDiscoverer`, `BestIntervalSubgroupDiscoverer`, `PRIMSubgroupDiscoverer`
- baselines (simple heuristics): `MORSSubgroupDiscoverer`, `RandomSubgroupDiscoverer`

All methods expect datasets with numeric feature values (as `pd.DataFrame`) and a binary prediction target (as `pd.Series`).
The heuristics are from literature or adaptations from other packages,
while we conceived the solver-based methods and baselines.
All subgroup-discovery methods can discover subgroups (who would have guessed?),
optionally using a limited number of features in the subgroup description (a feature-cardinality constraint),
while some of them can also find alternative subgroup descriptions.

Further, we provide four evaluation metrics for binary (`bool` or `int`) subgroup-membership vectors:

- `wracc()`: Weighted Relative Accuracy
- `nwracc()`: Weighted Relative Accuracy normalized to `[-1, 1]`
- `jaccard()`: Jaccary similarity (1 - Jaccard distance)
- `hamming()`: Normalized Hamming similarity (1 - Hamming distance normalized to `[0, 1]`; equals prediction accuracy)

For each of them, there is a general version (accepting `pd.Series`, `np.array`, or even simpler sequences like plain lists)
and a faster, `numpy`-specfic version with the same functionality (suffixed `_np`).

## Demo

We demonstrate subgroup discovery as well as alternative-subgroup-description discovery.

### Discovering (Original) Subgroups

Using a subgroup-discovery method from `csd` is similar to working with prediction models from `scikit-learn`.
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
`csd` also contains functions to evaluate predictions,
though the evaluation metrics for binary classification from `sklearn.metrics` should work as well.

Here is a small example for initialization, fitting, prediction, and evaluation:

```python
import csd
import sklearn.datasets
import sklearn.model_selection

X, y = sklearn.datasets.load_iris(as_frame=True, return_X_y=True)
y = (y == 1).astype(int)  # binary classification (in original data, three classes)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, train_size=0.8, random_state=25)

model = csd.BeamSearchSubgroupDiscoverer(beam_width=10)
fitting_result = model.fit(X=X_train, y=y_train)
print('Fitting result:', fitting_result)
print('Train WRAcc:', round(csd.wracc(y_true=y_train, y_pred=model.predict(X=X_train)), 2))
print('Test WRAcc:', round(csd.wracc(y_true=y_test, y_pred=model.predict(X=X_test)), 2))
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
import csd
import sklearn.datasets
import sklearn.model_selection

X, y = sklearn.datasets.load_iris(as_frame=True, return_X_y=True)
y = (y == 1).astype(int)  # binary classification (in original data, three classes)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, train_size=0.8, random_state=25)

model = csd.BeamSearchSubgroupDiscoverer(beam_width=10)
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
import csd
import sklearn.datasets
import sklearn.model_selection

X, y = sklearn.datasets.load_iris(as_frame=True, return_X_y=True)
y = (y == 1).astype(int)  # binary classification (in original data, three classes)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, train_size=0.8, random_state=25)

model = csd.BeamSearchSubgroupDiscoverer(beam_width=10, k=2, tau_abs=1, a=4)
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
For integration into the experimental pipeline of our paper, `fit()` should also return a dictionary
with the keys `objective_value`, `optimization_status` (may be `None`), and `optimization_time`.
If your subgroup-discovery method is not tailored towards a specific objective,
you may use the `wracc()` or `wracc_np()` from `csd` to guide the search
(the former is slower but supports more data types, the latter is tailored to `numpy` arrays and faster).

To support feature-cardinality constraints, like the existing subgroup-discovery methods,
you may want to allow setting an upper bound on the number of selected features `k`
during initialization and observing it during fitting.
We propose to store `k` in a field called `_k`;
however, it is not used in the methods of superclass `SubgroupDiscoverer`, as you need to implement the constraint during fitting.

If your subgroup-discovery method should also support the search for alternative subgroup descriptions,
make it a subclass of `AbstractSubgroupDiscoverer` (which inherits from `SubgroupDiscoverer`).
The search for alternatives and the fitting routine are already implemented there.
In particular, `fit()` dispatches to the method `_optimize()`, which you need to override.
The latter should both be able to

- search for the original subgroup, optimizing WRAcc or another notion of subgroup quality and
- search for alternative subgroup descriptions by
  - optimizing normalized Hamming similarity (see functions `hamming()` and `hamming_np` in `csd`)
    to the original subgroup (or another notion of subgroup similarity) and
  - having a constraint that at least `tau_abs` features from each previous subgroup description need to be de-selected
    (though not more features than actually were selected, to prevent infeasibilities).

Your implementation of `_optimize()` has to automatically switch the objective and add the constraints for alternatives if necessary.
The method's parameters `was_feature_selected_list` and `was_instance_in_box` tell you, if not `None`,
that you should search for an alternative subgroup description; else, search for the original subgroup.
Additionally, you should add parameters `a` and `tau_abs` in the initializer of your subgroup-discovery class,
storing them in the fields `_a` and `_tau_abs`.
The latter two fields are automatically used in `evaluate()` (to decide whether to search only for an original subgroup
or also for alternative subgroup descriptions) and should be used in your implementation of `_optimize()`.
