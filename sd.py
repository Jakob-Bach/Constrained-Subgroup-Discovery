"""Subgroup discovery

Classes for subgroup-discovery methods.
"""


from abc import ABCMeta, abstractmethod
import random
import time
from typing import Dict, Optional

import numpy as np
from ortools.linear_solver import pywraplp
import pandas as pd
import prelim.sd.BI
import z3


# Computes the weighted relative accuracy (WRAcc) for two binary (bool or int) series.
def wracc(y_true: pd.Series, y_pred: pd.Series) -> float:
    assert len(y_true) == len(y_pred), "Prediction and ground truth need to have same length."
    assert y_true.isin((0, 1, False, True)).all(), "Each ground-truth label needs to be binary."
    assert y_pred.isin((0, 1, False, True)).all(), "Each predicted label needs to be binary."
    n_true_pos = (y_true & y_pred).sum()
    n_instances = len(y_true)
    n_actual_pos = y_true.sum()
    n_pred_pos = y_pred.sum()
    return n_true_pos / n_instances - n_pred_pos * n_actual_pos / (n_instances ** 2)


# Same functionality as wracc(), but faster and intended for binary (bool or int) numpy arrays.
# Leaving out assertions cuts the execution time by roughly 50 % compared to the "slow" method,
# using numpy arrays instead of pandas Series by roughly an order of magnitude (not sure why).
# This fast method should be preferred if called often as a subroutine.
def wracc_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n_true_pos = np.count_nonzero(y_true & y_pred)
    n_instances = len(y_true)
    n_actual_pos = np.count_nonzero(y_true)
    n_pred_pos = np.count_nonzero(y_pred)
    return n_true_pos / n_instances - n_pred_pos * n_actual_pos / (n_instances ** 2)


class SubgroupDiscoverer(metaclass=ABCMeta):
    """Subgroup-discovery method

    The abstract base class for subgroup discovery. Defines a method signature for fitting, which
    needs to be overriden in subclasses, and a prediction method (similar to scikit-learn models).
    """

    # Initializes fields for lower bounds and upper bounds of the subgroup's box. These fields need
    # to be initialized in the fit() method (or the prediction method needs to be overridden).
    def __init__(self):
        self._box_lbs = None
        self._box_ubs = None

    # Should run the subgroup-discovery method on the given data, i.e., update self's internal
    # state appropriately for predictions later (e.g., set "self._box_lbs" and "self._box_ubs").
    # Should return meta-data about the fitting process, i.e., a dictionary with keys
    # "optimization_time" and "optimization_status".
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        raise NotImplementedError('Abstract method.')

    # Returns the lower bounds of the subgroup's box.
    def get_box_lbs(self) -> pd.Series:
        return self._box_lbs

    # Returns the upper bounds of the subgroup's box.
    def get_box_ubs(self) -> pd.Series:
        return self._box_ubs

    # Returns a series of predicted class labels (1 for instances in the box, else 0). Should only
    # be called after fit() since the box is undefined otherwise.
    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series((X.ge(self.get_box_lbs()) & X.le(self.get_box_ubs())).all(
            axis='columns').astype(int), index=X.index)

    # Same functionality as predict(), but faster and intended for numpy arrays instead of pandas
    # frames/series (as data types for the internally stored bounds as well as the passed data).
    def predict_np(self, X: np.ndarray) -> np.ndarray:
        prediction = np.ones(X.shape[0], dtype=bool)  # start by assuming each instance in box
        for j in range(X.shape[1]):
            prediction = prediction & (X[:, j] >= self._box_lbs[j]) & (X[:, j] <= self._box_ubs[j])
        return prediction

    # Trains, predicts, and evaluates subgroup discovery on a train-test split of a dataset.
    # Returns a dictionary with evaluation metrics.
    def evaluate(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame,
                 y_test: pd.Series) -> Dict[str, float]:
        start_time = time.process_time()
        results = self.fit(X=X_train, y=y_train)
        end_time = time.process_time()
        results['fitting_time'] = end_time - start_time
        results['train_wracc'] = wracc(y_true=y_train, y_pred=self.predict(X=X_train))
        results['test_wracc'] = wracc(y_true=y_test, y_pred=self.predict(X=X_test))
        return results


class MIPSubgroupDiscoverer(SubgroupDiscoverer):
    """MIP-based subgroup discovery method

    White-box formulation of subgroup discovery as a Mixed Integer Programming (MIP) optimization
    problem.
    """

    # Initialize fields. "timeout" should be indicated in seconds; if None, then no timeout.
    # "k" is the maximum number of features used in the subgroup description.
    def __init__(self, k: Optional[int] = None, timeout: Optional[float] = None):
        super().__init__()
        self._k = k
        self._timeout = timeout

    # Model and optimize subgroup discovery with Python-MIP. Return meta-data about the fitting
    # process (see superclass for more details).
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        assert y.isin((0, 1, False, True)).all(), 'Target "y" needs to be binary (bool or int).'

        # Define "speaking" names for certain constants in the optimization problem:
        n_instances = X.shape[0]
        n_features = X.shape[1]
        n_pos_instances = y.sum()
        feature_minima = X.min().to_list()
        feature_maxima = X.max().to_list()
        feature_diff_minima = X.apply(lambda col: pd.Series(col.sort_values().unique()).diff(
            ).min()).fillna(0).to_list()  # fillna() covers case that all values identical

        # Define optimizer:
        model = pywraplp.Solver.CreateSolver('SCIP')
        model.SetNumThreads(1)
        if self._timeout is not None:
            model.SetTimeLimit(round(self._timeout * 1000))  # int, measured in milliseconds

        # Define variables of the optimization problem:
        lb_vars = [model.NumVar(name=f'lb_{j}', lb=feature_minima[j], ub=feature_maxima[j])
                   for j in range(n_features)]
        ub_vars = [model.NumVar(name=f'ub_{j}', lb=feature_minima[j], ub=feature_maxima[j])
                   for j in range(n_features)]
        is_instance_in_box_vars = [model.BoolVar(name=f'x_{i}') for i in range(n_instances)]
        is_value_in_box_lb_vars = [[model.BoolVar(name=f'x_lb_{i}_{j}') for j in range(n_features)]
                                   for i in range(n_instances)]
        is_value_in_box_ub_vars = [[model.BoolVar(name=f'x_ub_{i}_{j}') for j in range(n_features)]
                                   for i in range(n_instances)]

        # Define auxiliary expressions for use in objective and potentially even constraints
        # (could also be variables, bound by "==" constraints; roughly same optimizer performance):
        n_instances_in_box = model.Sum(is_instance_in_box_vars)
        n_pos_instances_in_box = model.Sum([var for var, target in zip(is_instance_in_box_vars, y)
                                            if target == 1])

        # Define objective (WRAcc):
        model.Maximize(n_pos_instances_in_box / n_instances -
                       n_instances_in_box * n_pos_instances / (n_instances ** 2))

        # Define constraints:
        # (1) Identify for each instance if it is in the subgroup's box or not
        for i in range(n_instances):
            for j in range(n_features):
                # Approach for modeling constraint satisfaction: Binary variables (here:
                # "is_value_in_box_lb_vars[i][j]") indicate whether constraint satisfied
                # https://docs.mosek.com/modeling-cookbook/mio.html#constraint-satisfaction
                M = 2 * (feature_maxima[j] - feature_minima[j])  # large positive value
                m = 2 * (feature_minima[j] - feature_maxima[j])  # large (absolute) negative value
                model.Add(float(X.iloc[i, j]) + m * is_value_in_box_lb_vars[i][j]
                          <= lb_vars[j] - feature_diff_minima[j])  # get < rather than <=
                model.Add(lb_vars[j]
                          <= float(X.iloc[i, j]) + M * (1 - is_value_in_box_lb_vars[i][j]))
                model.Add(ub_vars[j] + m * is_value_in_box_ub_vars[i][j]
                          <= float(X.iloc[i, j]) - feature_diff_minima[j])
                model.Add(float(X.iloc[i, j])
                          <= ub_vars[j] + M * (1 - is_value_in_box_ub_vars[i][j]))
                # AND operator: https://docs.mosek.com/modeling-cookbook/mio.html#boolean-operators
                model.Add(is_instance_in_box_vars[i] <= is_value_in_box_lb_vars[i][j])
                model.Add(is_instance_in_box_vars[i] <= is_value_in_box_ub_vars[i][j])
                # third constraint for AND moved outside loop and summed up over features, since
                # only simultaneous satisfaction of all LB/UB constraints implies instance in box
            model.Add(model.Sum(is_value_in_box_lb_vars[i]) + model.Sum(is_value_in_box_ub_vars[i])
                      <= is_instance_in_box_vars[i] + 2 * n_features - 1)
        # (2) Relationship between lower and upper bound for each feature
        for j in range(n_features):
            model.Add(lb_vars[j] <= ub_vars[j])

        # (3) Limit number of features used in the box (i.e., where bounds exclude instances)
        if self._k is not None:
            is_feature_used_vars = [model.BoolVar(name=f'f_{j}') for j in range(n_features)]
            for j in range(n_features):
                # There is any Instance i where Feature j's value not in box -> Feature j used
                for i in range(n_instances):
                    model.Add(1 - is_value_in_box_lb_vars[i][j] <= is_feature_used_vars[j])
                    model.Add(1 - is_value_in_box_ub_vars[i][j] <= is_feature_used_vars[j])
                # Feature j used -> there is any Instance i where Feature j's value not in box
                model.Add(
                    is_feature_used_vars[j] <=
                    model.Sum(1 - is_value_in_box_lb_vars[i][j] for i in range(n_instances)) +
                    model.Sum(1 - is_value_in_box_ub_vars[i][j] for i in range(n_instances))
                )
            model.Add(model.Sum(is_feature_used_vars) <= self._k)

        # Optimize and store/return results:
        start_time = time.process_time()
        optimization_status = model.Solve()
        end_time = time.process_time()
        if optimization_status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            # Directly extracting values of "lb_vars" and "ub_vars" can lead to numeric issues
            # (e.g., instance from within box might fall slightly outside, even if all numerical
            # tolerances were set to 0 in optimization), so we use actual feature values of
            # instances in box instead; nice side-effect: box is tight around instances instead of
            # extending into margin around them
            is_instance_in_box = [bool(var.solution_value()) for var in is_instance_in_box_vars]
            self._box_lbs = X.iloc[is_instance_in_box].min()
            self._box_lbs[self._box_lbs == feature_minima] = float('-inf')
            self._box_ubs = X.iloc[is_instance_in_box].max()
            self._box_ubs[self._box_ubs == feature_maxima] = float('inf')
        else:
            self._box_lbs = pd.Series([float('-inf')] * X.shape[1], index=X.columns)
            self._box_ubs = pd.Series([float('inf')] * X.shape[1], index=X.columns)
        return {'optimization_status': optimization_status,
                'optimization_time': end_time - start_time}


class SMTSubgroupDiscoverer(SubgroupDiscoverer):
    """SMT-based subgroup-discovery method

    White-box formulation of subgroup discovery as a Satisfiability Modulo Theories (SMT)
    optimization problem.
    """

    # Initialize fields. "timeout" should be indicated in seconds; if None, then no timeout.
    # "k" is the maximum number of features used in the subgroup description.
    def __init__(self, k: Optional[int] = None, timeout: Optional[float] = None):
        super().__init__()
        self._k = k
        self._timeout = timeout

    # Model and optimize subgroup discovery with Z3. Return meta-data about the fitting process
    # (see superclass for more details).
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        assert y.isin((0, 1, False, True)).all(), 'Target "y" needs to be binary (bool or int).'

        # Define "speaking" names for certain constants in the optimization problem:
        n_instances = X.shape[0]
        n_features = X.shape[1]
        n_pos_instances = y.sum()
        feature_minima = X.min().to_list()
        feature_maxima = X.max().to_list()

        # Define variables of the optimization problem:
        lb_vars = [z3.Real(f'lb_{j}') for j in range(n_features)]
        ub_vars = [z3.Real(f'ub_{j}') for j in range(n_features)]
        is_instance_in_box_vars = [z3.Bool(f'x_{i}') for i in range(n_instances)]

        # Define auxiliary expressions for use in objective and potentially even constraints
        # (could also be variables, bound by "==" constraints; roughly same optimizer performance):
        n_instances_in_box = z3.Sum([z3.If(box_var, 1.0, 0) for box_var in is_instance_in_box_vars])
        n_pos_instances_in_box = z3.Sum([z3.If(box_var, 1.0, 0) for box_var, target
                                         in zip(is_instance_in_box_vars, y) if target == 1])

        # Define optimizer and objective (WRAcc):
        optimizer = z3.Optimize()
        if self._timeout is not None:
            # engine: see https://stackoverflow.com/questions/35203432/z3-minimization-and-timeout
            optimizer.set('maxsat_engine', 'wmax')
            optimizer.set('timeout', round(self._timeout * 1000))  # int, measured in milliseconds
        optimizer.maximize(n_pos_instances_in_box / n_instances -
                           n_instances_in_box * n_pos_instances / (n_instances ** 2))

        # Define constraints:
        # (1) Identify for each instance if it is in the subgroup's box or not
        for i in range(n_instances):
            optimizer.add(is_instance_in_box_vars[i] ==
                          z3.And([z3.And(float(X.iloc[i, j]) >= lb_vars[j],
                                         float(X.iloc[i, j]) <= ub_vars[j])
                                  for j in range(n_features)]))
        # (2) Relationship between lower and upper bound for each feature
        for j in range(n_features):
            optimizer.add(lb_vars[j] <= ub_vars[j])

        # (3) Limit number of features used in the box (i.e., where bounds exclude instances)
        if self._k is not None:
            optimizer.add(z3.Sum([z3.If(z3.Or(lb_vars[j] > feature_minima[j],
                                              ub_vars[j] < feature_maxima[j]), 1, 0)
                                  for j in range(n_features)]) <= self._k)

        # Optimize and store/return results:
        start_time = time.process_time()
        optimization_status = optimizer.check()
        end_time = time.process_time()
        # To avoid potential numeric issues when extracting values of real variables, use actual
        # feature values of instances in box as bounds (also makes box tight around instances).
        # If bounds do not exclude any instances (LB == min or UB == max) or if no valid model was
        # found (variables are None -> bool values are False -> no instances in box -> min()/max()
        # of box instances' feature values are NaN), use -/+ inf as bounds
        is_instance_in_box = [bool(optimizer.model()[var]) for var in is_instance_in_box_vars]
        self._box_lbs = X.iloc[is_instance_in_box].min().fillna(float('-inf'))
        self._box_lbs[self._box_lbs == feature_minima] = float('-inf')
        self._box_ubs = X.iloc[is_instance_in_box].max().fillna(float('inf'))
        self._box_ubs[self._box_ubs == feature_maxima] = float('inf')
        return {'optimization_status': str(optimization_status),
                'optimization_time': end_time - start_time}


class MORBSubgroupDiscoverer(SubgroupDiscoverer):
    """MORB (Minimal Optimal-Recall Box) baseline for subgroup discovery

    Choose the bounds as the minimum and maximum feature value of positive instances, so the box
    contains all positive instances and has the minimal size of all boxes doing so. Finds the
    optimal solution if a box exists that contains all positive and no negative instances.
    """

    # Initialize fields. "k" is the maximum number of features used in the subgroup description.
    def __init__(self, k: Optional[int] = None):
        super().__init__()
        self._k = k

    # Choose the minimal box that still has optimal recall (= 1). Return meta-data about the
    # fitting process (see superclass for more details).
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        assert y.isin((0, 1, False, True)).all(), 'Target "y" needs to be binary (bool or int).'
        # "Optimization": Find minima and maxima of positive instances:
        start_time = time.process_time()
        self._box_lbs = X[y == 1].min()
        self._box_ubs = X[y == 1].max()
        if (self._k is not None) and (self._k < X.shape[1]):
            # Count the number of false positives (negative instances in box) in each feature's
            # interval and reset the box for all features not in the bottom-k regardings FPs
            n_feature_fps = ((X[y == 0] >= self._box_lbs) & (X[y == 0] <= self._box_ubs)).sum()
            exclude_features = n_feature_fps.sort_values().index[self._k:]  # n-k highest
            self._box_lbs[exclude_features] = float('-inf')
            self._box_ubs[exclude_features] = float('inf')
        end_time = time.process_time()
        # Post-processing (as for optimizer-based solutions): if box extends to the limit of
        # feature values in the given data, treat this value as unbounded
        self._box_lbs[self._box_lbs == X.min()] = float('-inf')
        self._box_ubs[self._box_ubs == X.max()] = float('inf')
        return {'optimization_status': None,
                'optimization_time': end_time - start_time}


class RandomSubgroupDiscoverer(SubgroupDiscoverer):
    """Random-sampling baseline for subgroup discovery

    Choose the bounds repeatedly uniformly random from the unique values of each feature.
    """

    # Initialize fields. "k" is the maximum number of features used in the subgroup description.
    def __init__(self, k: Optional[int] = None, n_repetitions: int = 1000):
        super().__init__()
        self._k = k
        self._n_repetitions = n_repetitions

    # Repeatedly sample random boxes and pick the best of them. Return meta-data about the fitting
    # process (see superclass for more details).
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        assert y.isin((0, 1, False, True)).all(), 'Target "y" needs to be binary (bool or int).'
        unique_values_per_feature = [sorted(X[col].unique().tolist()) for col in X.columns]
        X_np = X.values  # working directly on numpy arrays rather than pandas faster
        y_np = y.values
        rng = random.Random(25)
        # "Optimization": Repeated random sampling
        start_time = time.process_time()
        opt_quality = float('-inf')
        for _ in range(self._n_repetitions):
            box_feature_idx = range(X.shape[1])  # default: all features may be restricted in box
            if (self._k is not None) and (self._k < X.shape[1]):
                box_feature_idx = rng.sample(box_feature_idx, k=self._k)
            # Sample two values per feature and "sort" them later into LB/UB; to sample uniformly
            # from constrained space, duplicate value from 1st sampling for 2nd sampling
            # (give "LB == UB" pairs two options to be sampled, as for any "LB != UB" pair)
            bounds1 = [rng.choice(unique_values_per_feature[j]) for j in box_feature_idx]
            bounds2 = [rng.choice(unique_values_per_feature[j] + [bounds1[box_j]])
                       for box_j, j in enumerate(box_feature_idx)]
            self._box_lbs = np.full(shape=X.shape[1], fill_value=-np.inf)  # made pd.Series later
            self._box_ubs = np.full(shape=X.shape[1], fill_value=np.inf)  # ... as numpy is faster
            self._box_lbs[box_feature_idx] = [min(b1, b2) for b1, b2 in zip(bounds1, bounds2)]
            self._box_ubs[box_feature_idx] = [max(b1, b2) for b1, b2 in zip(bounds1, bounds2)]
            quality = wracc_np(y_true=y_np, y_pred=self.predict_np(X=X_np))
            if quality > opt_quality:
                opt_quality = quality
                opt_box_lbs = self._box_lbs
                opt_box_ubs = self._box_ubs
        end_time = time.process_time()
        # Post-processing (as for optimizer-based solutions): if box extends to the limit of
        # feature values in the given data, treat this value as unbounded
        self._box_lbs = pd.Series(opt_box_lbs, index=X.columns)
        self._box_ubs = pd.Series(opt_box_ubs, index=X.columns)
        self._box_lbs[self._box_lbs == X.min()] = float('-inf')
        self._box_ubs[self._box_ubs == X.max()] = float('inf')
        return {'optimization_status': None,
                'optimization_time': end_time - start_time}


class BISubgroupDiscoverer(SubgroupDiscoverer):
    """Best-interval algorithm from the package "prelim"

    Heuristic search procedure using beam search.

    This wrapper class renames the parameter for the number of features used and allows it to be
    None (choosing it based on data dimensionality).

    Literature:
    - Mampaey et a. (2012): "Efficient Algorithms for Finding Richer Subgroup Descriptions in
      Numeric and Nominal Data"
    - Arzamasov et al. (2021): "REDS: Rule Extraction for Discovering Scenarios"
    """

    # Initialize fields. "k" is the maximum number of features used in the subgroup description.
    def __init__(self, k: Optional[int] = None):
        super().__init__()
        self._k = k

    # Run the algorithm from "prelim" with its default hyperparameters + the desired feature
    # cardinality. Return meta-data about the fitting process (see superclass for more details).
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        assert y.isin((0, 1, False, True)).all(), 'Target "y" needs to be binary (bool or int).'
        k = X.shape[1] if self._k is None else self._k
        model = prelim.sd.BI.BI(depth=k)  # parameter has a different name
        start_time = time.process_time()
        model.fit(X=X, y=y)
        end_time = time.process_time()
        # Unrestricted features of box are -/+ inf by default, so no separate initalization needed
        self._box_lbs = pd.Series(model.box_[0], index=X.columns)
        self._box_ubs = pd.Series(model.box_[1], index=X.columns)
        return {'optimization_status': None,
                'optimization_time': end_time - start_time}


class PRIMSubgroupDiscoverer(SubgroupDiscoverer):
    """PRIM algorithm

    Heuristic search procedure with a peeling phase (iteratively decreasing the range of the
    subgroup) and a pasting phase (iteratively increasing the range of the subgroup).
    In this version of the algorithm, only the peeling phase is implemented.
    Similar to the PRIM implementation in prelim.sd.PRIM, but has a different termination condition
    (min support instead of fixed iteration count combined with early-termination criterion),
    handles bounds differently (always produces strict bounds first but converts to <=/>= later)
    and supports a cardinality constraint on the number of restricted features.

    Literature:
    - Friedman & Fisher (1999): "Bump hunting in high-dimensional data"
    - https://github.com/Arzik1987/prelim/blob/main/src/prelim/sd/PRIM.py
    """

    # Initialize fields.
    # - "k" is the maximum number of features used in the subgroup description.
    # - "alpha" is the fraction of instances peeled off per iteration.
    # - "min_support" is the minimum fraction of instances in the box to continue peeling.
    def __init__(self, k: Optional[int] = None, alpha: float = 0.05, min_support: float = 0):
        super().__init__()
        self._k = k
        self._alpha = alpha
        self._min_support = min_support

    # Iteratively peel off (at least) a fraction "_alpha" of the instances by moving one bound of
    # one feature (choose best WRAcc); stop if empty box produced or "_min_support" violated.
    # Return meta-data about the fitting process (see superclass for more details).
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        assert y.isin((0, 1, False, True)).all(), 'Target "y" needs to be binary (bool or int).'
        X_np = X.values  # working directly on numpy arrays rather than pandas sometimes way faster
        y_np = y.values
        # Optimization: Iterative box updates
        start_time = time.process_time()
        self._box_lbs = X_np.min(axis=0)
        self._box_ubs = X_np.max(axis=0)
        y_pred = self.predict_np(X=X_np)
        opt_quality = wracc_np(y_true=y_np, y_pred=y_pred)
        opt_box_lbs = self._box_lbs.copy()  # fields will be changed for predictions, so copy
        opt_box_ubs = self._box_ubs.copy()
        has_peeled = True
        # Peeling continues as long as box has changed and contains certain number of instances
        while has_peeled and (np.count_nonzero(y_pred) / len(y_np) > self._min_support):
            # Note that peeling also changes "self._box_lbs" and "self._box_ubs"
            has_peeled = self._peel_one_step(X=X_np, y=y_np)
            y_pred = self.predict_np(X=X_np)
            quality = wracc_np(y_true=y_np, y_pred=y_pred)
            if quality > opt_quality:
                opt_quality = quality
                opt_box_lbs = self._box_lbs.copy()
                opt_box_ubs = self._box_ubs.copy()
        end_time = time.process_time()
        # Post-processing (as for optimizer-based solutions): if box extends to the limit of
        # feature values in the given data, treat this value as unbounded
        self._box_lbs = pd.Series(opt_box_lbs, index=X.columns)
        self._box_ubs = pd.Series(opt_box_ubs, index=X.columns)
        self._box_lbs[self._box_lbs == X.min()] = float('-inf')
        self._box_ubs[self._box_ubs == X.max()] = float('inf')
        return {'optimization_status': None,
                'optimization_time': end_time - start_time}

    # For each feature, check the "alpha" / "1 -  alpha" quantile for instances in the box as
    # potential new lower / upper bound. Choose the feature and bound with the best objective
    # (WRAcc) value. Return if peeling was successful (fails if only an emtpy box would be
    # created, e.g., if features only contain one value each or empty box is optimal).
    def _peel_one_step(self, X: np.ndarray, y: np.ndarray) -> bool:
        is_instance_in_old_box = self.predict_np(X=X)
        opt_quality = float('-inf')  # select one peel even if it's not better than previous box
        opt_feature_idx = None
        opt_bound = None
        opt_is_ub = None  # either LB or UB updated
        # Ensure feature-cardinality constraint:
        if self._k is None:
            candidate_feature_idxs = range(X.shape[1])
        else:
            used_feature_idxs = np.where(((X < self._box_lbs) |
                                          (X > self._box_ubs)).any(axis=0))[0]
            if len(used_feature_idxs) == self._k:
                candidate_feature_idxs = used_feature_idxs
            elif len(used_feature_idxs) > self._k:
                raise RuntimeError('The algorithm used more features than allowed.')
            else:
                candidate_feature_idxs = range(X.shape[1])
        # Exclude features only having one unique value (i.e., all feature values equal first one):
        candidate_feature_idxs = [j for j in candidate_feature_idxs if (X[:, j] != X[0, j]).any()]
        if len(candidate_feature_idxs) == 0:  # no peeling possible
            return False  # box unchanged
        for j in candidate_feature_idxs:
            # Check a new lower bound (if quantile between two feature values, choose middle):
            bound = np.quantile(X[is_instance_in_old_box, j], q=self._alpha, method='midpoint')
            # Only checking the new bound and combining with prior information (on whether instance
            # in box) is faster than updating self._box_lbs and using predict_np();
            # also, using strict equality (as in original paper) here, will be made >= later
            y_pred = is_instance_in_old_box & (X[:, j] > bound)
            quality = wracc_np(y_true=y, y_pred=y_pred)
            if quality > opt_quality:
                opt_quality = quality
                opt_feature_idx = j
                opt_bound = bound
                opt_is_ub = False
            # Check a new upper bound (if quantile between two feature values, choose middle):
            bound = np.quantile(X[is_instance_in_old_box, j], q=1-self._alpha, method='midpoint')
            y_pred = is_instance_in_old_box & (X[:, j] < bound)
            quality = wracc_np(y_true=y, y_pred=y_pred)
            if quality > opt_quality:
                opt_quality = quality
                opt_feature_idx = j
                opt_bound = bound
                opt_is_ub = True
        if opt_is_ub:
            # Convert ">", potentially for a midpoint value, to ">=" for an actual feature value:
            in_box_values = X[is_instance_in_old_box & (X[:, opt_feature_idx] < opt_bound),
                              opt_feature_idx]
            if len(in_box_values) > 0:
                self._box_ubs[opt_feature_idx] = float(in_box_values.max())
        else:
            in_box_values = X[is_instance_in_old_box & (X[:, opt_feature_idx] > opt_bound),
                              opt_feature_idx]
            if len(in_box_values) > 0:
                self._box_lbs[opt_feature_idx] = float(in_box_values.min())
        return len(in_box_values) > 0  # New, non-empty box produced


class BeamSearchSubgroupDiscoverer(SubgroupDiscoverer):
    """Beam-search algorithm

    Heuristic search procedure that maintains a beam (list) of candidate boxes and iteratively
    refines them, each iteration testing all possible changes of lower and upper bounds per
    candidate box (but only one change at a time); retains a certain (beam width) number of boxes
    with the highest quality.
    Inspired by the beam-search implementation in pysubgroup.BeamSearch, but faster and supports
    a cardinality constraint on the number of restricted features.

    Literature: https://github.com/flemmerich/pysubgroup/blob/master/src/pysubgroup/algorithms.py
    """

    # Initialize fields.
    # - "k" is the maximum number of features used in the subgroup description.
    # - "beam_width" is the number of candidate subgroups kept per iteration (lower == faster but
    #   potentially lower quality).
    def __init__(self, k: Optional[int] = None, beam_width: int = 10):
        super().__init__()
        self._k = k
        self._beam_width = beam_width

    # Iteratively refine boxes; stop if no box in the beam has changed in the previous iteration.
    # Return meta-data about the fitting process (see superclass for more details).
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        assert y.isin((0, 1, False, True)).all(), 'Target "y" needs to be binary (bool or int).'
        unique_values_per_feature = [sorted(X[col].unique()) for col in X.columns]
        X_np = X.values  # working directly on numpy arrays rather than pandas faster
        y_np = y.values
        start_time = time.process_time()
        # Initially, all boxes in the beam are unbounded, i.e., limits are (-inf, inf) per feature:
        beam_bounds = np.array([[np.repeat(-np.inf, repeats=X_np.shape[1]),
                                 np.repeat(np.inf, repeats=X_np.shape[1])]
                                for _ in range(self._beam_width)])  # beam width * 2 * num features
        # Indices (regarding feature's unique values) for these bounds are 0 and the max index:
        beam_bound_idxs = np.array([[np.zeros(shape=X_np.shape[1], dtype=int),
                                     [len(x) - 1 for x in unique_values_per_feature]]
                                    for _ in range(self._beam_width)])  # beam width * 2 * num feat
        # Initial boxes contain all instances:
        beam_is_in_box = np.array([np.ones(shape=X_np.shape[0], dtype=bool)
                                   for _ in range(self._beam_width)])  # beam width * num instances
        # All boxes should be considered for updates:
        cand_has_changed = np.ones(shape=self._beam_width, dtype=bool)
        # Boxes containing all instances have WRAcc of 0:
        cand_quality = np.zeros(shape=self._beam_width)
        cand_min_quality = 0
        while np.count_nonzero(cand_has_changed) > 0:  # at least one box changed last iteration
            # Copy boxes from the beam since the candidates for next beam will be updated, but we
            # still want to iterate over the unchanged beam of the previous iteration:
            cand_bounds = beam_bounds.copy()
            cand_bound_idxs = beam_bound_idxs.copy()
            cand_is_in_box = beam_is_in_box.copy()
            prev_cand_changed_idxs = np.where(cand_has_changed)[0]
            cand_has_changed = np.zeros(shape=self._beam_width, dtype=bool)
            # Iterate over all previously updated boxes in the beam and try to refine them:
            for box_idx in prev_cand_changed_idxs:
                bounds = beam_bounds[box_idx]
                bound_idxs = beam_bound_idxs[box_idx]
                is_in_box = beam_is_in_box[box_idx]
                # If feature-cardinality constraint used, check how many features restricted in
                # boy; if already k, only consider these features; else, consider all features:
                if self._k is None:
                    candidate_feature_idxs = range(X_np.shape[1])
                else:
                    used_feature_idxs = np.where(((X_np < bounds[0]) |
                                                  (X_np > bounds[1])).any(axis=0))[0]
                    if len(used_feature_idxs) == self._k:
                        candidate_feature_idxs = used_feature_idxs
                    elif len(used_feature_idxs) > self._k:
                        raise RuntimeError('The algorithm used more features than allowed.')
                    else:
                        candidate_feature_idxs = range(X_np.shape[1])
                for j in candidate_feature_idxs:
                    # Test new values for lower bound (> box's previous one but <= box's UB)
                    for value_idx in range(bound_idxs[0, j] + 1, bound_idxs[1, j] + 1):
                        bound_value = unique_values_per_feature[j][value_idx]
                        y_pred = is_in_box & (X_np[:, j] >= bound_value)
                        quality = wracc_np(y_true=y_np, y_pred=y_pred)
                        # Replace the minimum-quality candidate, but only if newly created box not
                        # already a candidate (same new box may be created from multiple old boxes)
                        if quality > cand_min_quality:
                            new_bounds = bounds.copy()
                            new_bounds[0, j] = bound_value
                            if all((x != new_bounds).any() for x in cand_bounds):
                                min_quality_idx = np.where(cand_quality == cand_min_quality)[0][0]
                                cand_bounds[min_quality_idx] = new_bounds
                                cand_bound_idxs[min_quality_idx] = bound_idxs.copy()
                                cand_bound_idxs[min_quality_idx, 0, j] = value_idx
                                cand_is_in_box[min_quality_idx] = y_pred
                                cand_has_changed[min_quality_idx] = True
                                cand_quality[min_quality_idx] = quality
                                cand_min_quality = cand_quality.min()
                    # Test new values for upper bound (< box's previous one but >= box's LB)
                    for value_idx in range(bound_idxs[0, j], bound_idxs[1, j]):
                        bound_value = unique_values_per_feature[j][value_idx]
                        y_pred = is_in_box & (X_np[:, j] <= bound_value)
                        quality = wracc_np(y_true=y_np, y_pred=y_pred)
                        if quality > cand_min_quality:
                            new_bounds = bounds.copy()
                            new_bounds[1, j] = bound_value
                            if all((x != new_bounds).any() for x in cand_bounds):
                                min_quality_idx = np.where(cand_quality == cand_min_quality)[0][0]
                                cand_bounds[min_quality_idx] = new_bounds
                                cand_bound_idxs[min_quality_idx] = bound_idxs.copy()
                                cand_bound_idxs[min_quality_idx, 1, j] = value_idx
                                cand_is_in_box[min_quality_idx] = y_pred
                                cand_has_changed[min_quality_idx] = True
                                cand_quality[min_quality_idx] = quality
                                cand_min_quality = cand_quality.min()
            # Best candidates are next beam; copy() unnecessary, as next candidates will be copied:
            beam_bounds = cand_bounds
            beam_bound_idxs = cand_bound_idxs
            beam_is_in_box = cand_is_in_box
        # Select best subgroup out of beam:
        max_quality_idx = np.where(cand_quality == cand_quality.max())[0][0]
        end_time = time.process_time()
        # Post-processing (as for optimizer-based solutions): if box extends to the limit of
        # feature values in the given data, treat this value as unbounded
        self._box_lbs = pd.Series(beam_bounds[max_quality_idx, 0], index=X.columns)
        self._box_ubs = pd.Series(beam_bounds[max_quality_idx, 1], index=X.columns)
        self._box_lbs[self._box_lbs == X.min()] = float('-inf')
        self._box_ubs[self._box_ubs == X.max()] = float('inf')
        return {'optimization_status': None,
                'optimization_time': end_time - start_time}
