"""Subgroup-discovery methods

Classes for subgroup-discovery methods.
"""


from abc import ABCMeta, abstractmethod
import random
import time
from typing import Any, Dict, Optional, Sequence

import numpy as np
from ortools.linear_solver import pywraplp
import pandas as pd
import z3

from .metrics import nwracc, hamming, hamming_np, jaccard, wracc, wracc_np


class SubgroupDiscoverer(metaclass=ABCMeta):
    """Subgroup-discovery method

    The abstract base class for subgroup-discovery methods. Defines an abstract method for fitting,
    which needs to be overridden in subclasses, and a prediction method, which can be used as-is
    (if fit() is implemented properly). Thus, the interface is similar to scikit-learn models.
    """

    # Initialize fields for lower bounds and upper bounds of the subgroup's box. These fields need
    # to be initialized in the fit() method (or the prediction method needs to be overridden).
    def __init__(self):
        self._box_lbs = None
        self._box_ubs = None

    # Should run the subgroup-discovery method on the given data. In particular, update self's
    # internal state appropriately for predictions later by setting the fields "self._box_lbs" and
    # "self._box_ubs" as pd.Series; if a feature is unbounded, -/+ inf should be used as bounds.
    # Should return meta-data about the fitting process, i.e., a dictionary with keys
    # "objective_value", "optimization_time", and "optimization_status".
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        raise NotImplementedError('Abstract method.')

    # Return the lower bounds of the subgroup's box.
    def get_box_lbs(self) -> pd.Series:
        return self._box_lbs

    # Return the upper bounds of the subgroup's box.
    def get_box_ubs(self) -> pd.Series:
        return self._box_ubs

    # Return a binary sequence indicating for each feature if selected (restricted) in subgroup.
    def is_feature_selected(self) -> Sequence[bool]:
        return ((self.get_box_lbs() != float('-inf')) |
                (self.get_box_ubs() != float('inf'))).to_list()

    # Return the indices of features selected (restricted) in subgroup.
    def get_selected_feature_idxs(self) -> Sequence[int]:
        return [j for j, is_selected in enumerate(self.is_feature_selected()) if is_selected]

    # Return a pd.Series of predicted class labels (1 for instances in the box, else 0).
    # Should only be called after fit() since the box is undefined otherwise.
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

    # Train, predict, and evaluate the subgroup-discovery method on a train-test split.
    # Return a data frame with evaluation metrics and bounds on features as columns. Each row
    # represents a subgroup; unless this method is overridden, there will be only one row/subgroup.
    def evaluate(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame,
                 y_test: pd.Series) -> pd.DataFrame:
        start_time = time.process_time()
        results = self.fit(X=X_train, y=y_train)  # returns a dict with evaluation metrics
        end_time = time.process_time()
        y_pred_train = self.predict(X=X_train)
        y_pred_test = self.predict(X=X_test)
        results['fitting_time'] = end_time - start_time
        results['train_wracc'] = wracc(y_true=y_train, y_pred=y_pred_train)
        results['test_wracc'] = wracc(y_true=y_test, y_pred=y_pred_test)
        results['train_nwracc'] = nwracc(y_true=y_train, y_pred=y_pred_train)
        results['test_nwracc'] = nwracc(y_true=y_test, y_pred=y_pred_test)
        results['box_lbs'] = self.get_box_lbs().tolist()
        results['box_ubs'] = self.get_box_ubs().tolist()
        results['selected_feature_idxs'] = self.get_selected_feature_idxs()
        # Convert dict into single-row DataFrame; subclasses may return multiple subgroups (= rows)
        return pd.DataFrame([results])


class AlternativeSubgroupDiscoverer(SubgroupDiscoverer, metaclass=ABCMeta):
    """Subgroup-discovery method supporting alternatives

    The abstract base class for subgroup-discovery methods that cannot only find one original
    subgroup but also alternative descriptions for the original one. Implements the (formerly
    abstract) fit() method by dispatching to a new abstract optimization method, which should be
    overridden to (based on the passed parameters) either find an orignal subgroup or alternative
    descriptions. evaluate() is also overridden to either perform the evaluate() routine from the
    superclass (fit and evaluate original subgroup) or search for original subgroup + alternative
    descriptions, dispatching to a generic search routine that calls the new optimization method.
    In particular, if _optimize() is implemented properly, this search routine for alternative
    descriptions should also work in subclasses.
    """

    # Initialize fields.
    # - "a" is the number of alternative subgroup descriptions; if None, then none (only original
    #   subgroup searched). Should only be set if feature cardinality "k" is set and processed in
    #   subclass as well (else there may be no alternatives).
    # - "tau_abs" is the number of features selected in each existing subgroup description that
    #   should *not* be selected in alternative subgroup description; parameter should be set if
    #   and only if "a" is set.
    def __init__(self, a: Optional[int] = None, tau_abs: Optional[int] = None):
        super().__init__()
        self._a = a
        self._tau_abs = tau_abs

    # Should either find original subgroup (if only "X" and "y" are passed) or an alternative
    # subgroup description (if the optional arguments are passed). Should update self's internal
    # state appropriately for predictions later (i.e., set "self._box_lbs" and "self._box_ubs").
    # Should Return meta-data about the optimization process, i.e., a dictionary with keys
    # "objective_value", "optimization_time", and "optimization_status" (like fit() does).
    # - "was_feature_selected_list": for each existing subgroup description and each feature,
    #   indicate if selected. An alternative subgroup description should deselect (*not* select)
    #   at least "self._tau_abs" features from each existing subgroup description.
    # - "was_instance_in_box": for each instance, indicate if in original subgroup. An alternative
    #   subgroup should try to maximize the normalized Hamming similarity to this prediction.
    @abstractmethod
    def _optimize(self, X: pd.DataFrame, y: pd.Series,
                  was_feature_selected_list: Optional[Sequence[Sequence[bool]]] = None,
                  was_instance_in_box: Optional[Sequence[bool]] = None) -> Dict[str, Any]:
        raise NotImplementedError('Abstract method.')

    # Run subgroup-discovery method. Return meta-data about the fitting process (see superclass for
    # more details).
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        # Dispatch to another, more general routine (which can also find alternative subgroup
        # descriptions; here, consistent to fit() in other classes, only one subgroup searched):
        return self._optimize(X=X, y=y, was_feature_selected_list=None, was_instance_in_box=None)

    # Train, predict, and evaluate the subgroup-discovery method multiple times on a train-test
    # split of a dataset. Each of the "self._a" alternative subgroup descriptions should deselect
    # (*not* use) at least "self._tau_abs" features from each existing (previous) description.
    # The original subgroup should optimize WRAcc or another measure of subgroup quality (subject
    # to implementation of "_optimize()"), all subsequent subgroups should optimize normalized
    # Hamming similarity to the original one or another measure of subgroup similarity.
    # Returns a data frame with evaluation metrics, each row corresponding to a subgroup.
    def search_alternative_descriptions(self, X_train: pd.DataFrame, y_train: pd.Series,
                                        X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        was_feature_selected_list = []  # i-th entry corresponds to i-th subgroup (List[bool])
        was_instance_in_box = None  # is instance in 0-th subgroup? (List[Bool])
        results = []
        for i in range(self._a + 1):
            start_time = time.process_time()
            if i == 0:  # original subgroup, whose prediction should be replicated by alternatives
                result = self._optimize(X=X_train, y=y_train)  # dict with evaluation metrics
            else:
                result = self._optimize(X=X_train, y=y_train,
                                        was_feature_selected_list=was_feature_selected_list,
                                        was_instance_in_box=was_instance_in_box)
            end_time = time.process_time()
            y_pred_train = self.predict(X=X_train)
            y_pred_test = self.predict(X=X_test)
            if i == 0:
                was_instance_in_box = y_pred_train.astype(bool).to_list()
            was_feature_selected_list.append(self.is_feature_selected())
            result['fitting_time'] = end_time - start_time
            result['train_wracc'] = wracc(y_true=y_train, y_pred=y_pred_train)
            result['test_wracc'] = wracc(y_true=y_test, y_pred=y_pred_test)
            result['train_nwracc'] = nwracc(y_true=y_train, y_pred=y_pred_train)
            result['test_nwracc'] = nwracc(y_true=y_test, y_pred=y_pred_test)
            result['alt.hamming'] = hamming(sequence_1=was_instance_in_box,
                                            sequence_2=y_pred_train)
            result['alt.jaccard'] = jaccard(set_1_indicators=was_instance_in_box,
                                            set_2_indicators=y_pred_train)
            result['alt.number'] = i
            result['box_lbs'] = self.get_box_lbs().tolist()
            result['box_ubs'] = self.get_box_ubs().tolist()
            result['selected_feature_idxs'] = self.get_selected_feature_idxs()
            results.append(result)
        return pd.DataFrame(results)

    # Train, predict, and evaluate the subgroup-discovery method on a train-test split of a
    # dataset. Return a data frame with evaluation metrics and bounds on features. If the fields
    # for alternatives are set ("_a" and "_tau_abs"), multiple subgroup descriptions are found
    # each forming a row in the results), else only one (as is the default for the superclass).
    def evaluate(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame,
                 y_test: pd.Series) -> pd.DataFrame:
        if (self._a is not None) and (self._tau_abs is not None):
            return self.search_alternative_descriptions(X_train=X_train, y_train=y_train,
                                                        X_test=X_test, y_test=y_test)
        else:
            return super().evaluate(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)


class MIPSubgroupDiscoverer(SubgroupDiscoverer):
    """MIP-based subgroup-discovery method

    White-box formulation of subgroup discovery as a Mixed Integer (Linear) Programming (MIP)
    optimization problem, which is tackled by a solver (SCIP via the package "mip").
    """

    # Initialize fields.
    # - "timeout" should be indicated in seconds; if None, then no timeout.
    # -  "k" is the maximum number of features that may be selected in the subgroup description;
    #   if None, then all.
    def __init__(self, k: Optional[int] = None, timeout: Optional[float] = None):
        super().__init__()
        self._k = k
        self._timeout = timeout

    # Model and optimize subgroup discovery with Python-MIP. Return meta-data about the fitting
    # process (see superclass for more details). The optimization status is 0 for the optimum
    # without timeout, 1 for any (potentially suboptimal) solution (-> timeout), and 6 if no
    # solution found yet (-> timeout).
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        assert y.isin((0, 1, False, True)).all(), 'Target "y" needs to be binary (bool or int).'

        # Define "speaking" names for certain constants in the optimization problem:
        n_instances = X.shape[0]
        n_features = X.shape[1]
        n_pos_instances = y.sum()
        feature_minima = X.min().to_list()
        feature_maxima = X.max().to_list()
        feature_diff_minima = X.apply(lambda col: pd.Series(col.sort_values().unique()).diff(
            ).min()).fillna(0).to_list()
        # fillna() covers the case that all values are identical, for which we want the the strict
        # inequalities to become non-strict (else optimization problem infeasible since
        # inequalities would enforce UB < LB)

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
        is_instance_in_box_vars = [model.BoolVar(name=f'b_{i}') for i in range(n_instances)]
        is_value_in_box_lb_vars = [[model.BoolVar(name=f'b_lb_{i}_{j}') for j in range(n_features)]
                                   for i in range(n_instances)]
        is_value_in_box_ub_vars = [[model.BoolVar(name=f'b_ub_{i}_{j}') for j in range(n_features)]
                                   for i in range(n_instances)]
        is_feature_selected_vars = [model.BoolVar(name=f's_{j}') for j in range(n_features)]
        is_feature_selected_lb_vars = [model.BoolVar(name=f's_lb_{j}') for j in range(n_features)]
        is_feature_selected_ub_vars = [model.BoolVar(name=f's_ub_{j}') for j in range(n_features)]

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
                # Approach for modeling constraint satisfaction: Binary variables
                # "is_value_in_box_lb_vars[i][j]" and "is_value_in_box_ub_vars[i][j]" indicate
                # whether constraints "lb_j <= X_{ij}" and "X_{ij} <= ub_j"  satisfied
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
        # (3) Limit number of features selected in the box (i.e., where bounds exclude instances)
        for j in range(n_features):
            # There is any Instance i where Feature j's value not in box -> Feature j selected
            for i in range(n_instances):
                model.Add(1 - is_value_in_box_lb_vars[i][j] <= is_feature_selected_lb_vars[j])
                model.Add(1 - is_value_in_box_ub_vars[i][j] <= is_feature_selected_ub_vars[j])
            model.Add(is_feature_selected_lb_vars[j] <= is_feature_selected_vars[j])
            model.Add(is_feature_selected_ub_vars[j] <= is_feature_selected_vars[j])
            # Feature j selected -> there is any Instance i where Feature j's value not in box
            model.Add(
                is_feature_selected_vars[j] <=
                model.Sum(1 - is_value_in_box_lb_vars[i][j] for i in range(n_instances)) +
                model.Sum(1 - is_value_in_box_ub_vars[i][j] for i in range(n_instances))
            )
        if self._k is not None:
            model.Add(model.Sum(is_feature_selected_vars) <= self._k)

        # Optimize and store/return results:
        start_time = time.process_time()
        optimization_status = model.Solve()
        end_time = time.process_time()
        if optimization_status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            # Directly extracting values of "lb_vars" and "ub_vars" can lead to numeric issues
            # (e.g., instance from within box might fall slightly outside, even if all numerical
            # tolerances were set to 0 in optimization), so we use actual feature values of
            # instances in box instead; nice side-effect: box is tight around instances instead of
            # extending into the margin around them.
            # If the box is empty, use +inf as LB and -inf as UB for selected features.
            # For features where lower/upper bounds do not exclude any instance, use -/+ inf.
            is_instance_in_box = [bool(var.solution_value()) for var in is_instance_in_box_vars]
            is_lb_unused = [not bool(var.solution_value()) for var in is_feature_selected_lb_vars]
            is_ub_unused = [not bool(var.solution_value()) for var in is_feature_selected_ub_vars]
            if any(is_instance_in_box):
                self._box_lbs = X.iloc[is_instance_in_box].min()
                self._box_ubs = X.iloc[is_instance_in_box].max()
            else:  # min()/max() would yield NaN; use invalid bounds instead
                self._box_lbs = pd.Series([float('inf')] * X.shape[1], index=X.columns)
                self._box_ubs = pd.Series([float('-inf')] * X.shape[1], index=X.columns)
            self._box_lbs[is_lb_unused] = float('-inf')
            self._box_ubs[is_ub_unused] = float('inf')
            objective_value = model.Objective().Value()
        else:
            self._box_lbs = pd.Series([float('-inf')] * X.shape[1], index=X.columns)
            self._box_ubs = pd.Series([float('inf')] * X.shape[1], index=X.columns)
            objective_value = float('nan')
        return {'objective_value': objective_value,
                'optimization_status': optimization_status,
                'optimization_time': end_time - start_time}


class SMTSubgroupDiscoverer(AlternativeSubgroupDiscoverer):
    """SMT-based subgroup-discovery method

    White-box formulation of subgroup discovery as a Satisfiability Modulo Theories (SMT)
    optimization problem, which is tackled by a solver (Z3 via the package "z3").
    """

    # Initialize fields.
    # - "timeout" should be indicated in seconds; if None, then no timeout.
    # - "k" is the maximum number of features that may selected in the subgroup description;
    #   if None, then all.
    # - "a" is the number of alternative subgroup descriptions; if None, then none (only original
    #   subgroup searched). Should only be set if "k" is set (else there may be no alternatives).
    # - "tau_abs" is the number of features selected in each existing subgroup description that
    #   should be deselected (*not* be selected) in alternative subgroup description; parameter
    #   should be set if and only if "a" is set.
    def __init__(self, timeout: Optional[float] = None, k: Optional[int] = None,
                 a: Optional[int] = None, tau_abs: Optional[int] = None):
        super().__init__(a=a, tau_abs=tau_abs)
        self._timeout = timeout
        self._k = k

    # Model and optimize subgroup discovery with Z3. Either find original subgroup (if only "X" and
    # "y" are passed) or an alternative subgroup description (if optional arguments are passed).
    # Return meta-data about the optimization process (see superclass for details).
    # The optimization status is "sat" if the optimum was found and "unknown" if not (-> timeout).
    def _optimize(self, X: pd.DataFrame, y: pd.Series,
                  was_feature_selected_list: Optional[Sequence[Sequence[bool]]] = None,
                  was_instance_in_box: Optional[Sequence[bool]] = None) -> Dict[str, Any]:
        assert y.isin((0, 1, False, True)).all(), 'Target "y" needs to be binary (bool or int).'
        is_alternative = was_instance_in_box is not None

        # Define "speaking" names for certain constants in the optimization problem:
        n_instances = X.shape[0]
        n_features = X.shape[1]
        n_pos_instances = y.sum()
        feature_minima = X.min().to_list()
        feature_maxima = X.max().to_list()

        # Define variables of the optimization problem:
        lb_vars = [z3.Real(f'lb_{j}') for j in range(n_features)]
        ub_vars = [z3.Real(f'ub_{j}') for j in range(n_features)]
        is_instance_in_box_vars = [z3.Bool(f'b_{i}') for i in range(n_instances)]
        is_feature_selected_vars = [z3.Bool(f's_{j}') for j in range(n_features)]
        is_feature_selected_lb_vars = [z3.Bool(f's_lb_{j}') for j in range(n_features)]
        is_feature_selected_ub_vars = [z3.Bool(f's_ub_{j}') for j in range(n_features)]

        # Define optimizer and objective:
        optimizer = z3.Optimize()
        if self._timeout is not None:
            # engine: see https://stackoverflow.com/questions/35203432/z3-minimization-and-timeout
            optimizer.set('maxsat_engine', 'wmax')
            optimizer.set('timeout', round(self._timeout * 1000))  # int, measured in milliseconds
        if is_alternative:
            # Optimize Hamming similarity to previous instance-in-box values, normalized by total
            # number of instances ("* 1.0" enforces float operations (else int)):
            objective = optimizer.maximize(
                z3.Sum([var if val else z3.Not(var) for var, val
                        in zip(is_instance_in_box_vars, was_instance_in_box)]) * 1.0 / n_instances)
        else:
            # Optimize WRAcc; define two auxiliary expressions first, which could also be variables
            # bound by "==" constraints (roughly same optimizer performance):
            n_instances_in_box = z3.Sum([z3.If(box_var, 1.0, 0)
                                         for box_var in is_instance_in_box_vars])
            n_pos_instances_in_box = z3.Sum([z3.If(box_var, 1.0, 0) for box_var, target
                                             in zip(is_instance_in_box_vars, y) if target == 1])
            objective = optimizer.maximize(
                n_pos_instances_in_box / n_instances -
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
        # (3) Limit number of features selected in the box (i.e., where bounds exclude instances)
        for j in range(n_features):
            optimizer.add(is_feature_selected_lb_vars[j] == (lb_vars[j] > feature_minima[j]))
            optimizer.add(is_feature_selected_ub_vars[j] == (ub_vars[j] < feature_maxima[j]))
            optimizer.add(is_feature_selected_vars[j] == z3.Or(is_feature_selected_lb_vars[j],
                                                               is_feature_selected_ub_vars[j]))
        if self._k is not None:
            optimizer.add(z3.Sum(is_feature_selected_vars) <= self._k)
        # (4) Make alternatives not select a certain number of features selected in other subgroups
        if is_alternative:
            for was_feature_selected in was_feature_selected_list:
                k_used = sum(was_feature_selected)  # may be smaller than prescribed "self._k"
                optimizer.add(
                    z3.Sum([z3.Not(is_feature_selected_vars[j]) for j in range(n_features)
                            if was_feature_selected[j]]) >= min(self._tau_abs, k_used)
                )

        # Optimize:
        start_time = time.process_time()
        optimization_status = optimizer.check()
        end_time = time.process_time()

        # Prepare and return results:
        if objective.value().is_int():  # type IntNumRef
            objective_value = float(objective.value().as_long())
        else:  # type RatNumRef
            objective_value = float(objective.value().numerator_as_long() /
                                    objective.value().denominator_as_long())
        # To avoid potential numeric issues when extracting values of real variables, use actual
        # feature values of instances in box as bounds (also makes box tight around instances).
        # If the box is empty, use +inf as LB and -inf as UB for selected features.
        # For features where lower/upper bounds do not exclude any instance, or if no valid model
        # was found (variables are None -> bool values are False), use -/+ inf as bounds.
        is_instance_in_box = [bool(optimizer.model()[var]) for var in is_instance_in_box_vars]
        is_lb_unused = [not bool(optimizer.model()[var]) for var in is_feature_selected_lb_vars]
        is_ub_unused = [not bool(optimizer.model()[var]) for var in is_feature_selected_ub_vars]
        if any(is_instance_in_box):
            self._box_lbs = X.iloc[is_instance_in_box].min()
            self._box_ubs = X.iloc[is_instance_in_box].max()
        else:  # min()/max() would yield NaN; use invalid bounds instead
            self._box_lbs = pd.Series([float('inf')] * X.shape[1], index=X.columns)
            self._box_ubs = pd.Series([float('-inf')] * X.shape[1], index=X.columns)
        self._box_lbs.iloc[is_lb_unused] = float('-inf')
        self._box_ubs.iloc[is_ub_unused] = float('inf')
        return {'objective_value': objective_value,
                'optimization_status': str(optimization_status),
                'optimization_time': end_time - start_time}


class MORSSubgroupDiscoverer(SubgroupDiscoverer):
    """MORS (Minimal Optimal-Recall Subgroup) baseline for subgroup discovery

    Choose the bounds as the minimum and maximum feature value of positive instances, so the box

    - contains all positive instances (i.e., has optimal recall) and
    - has the minimal size of all boxes doing so, i.e., minimizes the number of false positives
      (or, equivalently, maximizes the number of true negatives).

    Finds a perfect subgroup (box containing all positive and no negative instances) if it exists.
    """

    # Initialize fields. "k" is the maximum number of features that may be selected in the subgroup
    # description.
    def __init__(self, k: Optional[int] = None):
        super().__init__()
        self._k = k

    # Choose the minimal box that still has optimal recall (= 1). Return meta-data about the
    # fitting process (see superclass for more details); does not explicitly compute its objective
    # value, therefore "None" returned for this metric.
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        assert y.isin((0, 1, False, True)).all(), 'Target "y" needs to be binary (bool or int).'
        # "Optimization": Find minima and maxima of positive instances:
        start_time = time.process_time()
        self._box_lbs = X[y == 1].min()
        self._box_ubs = X[y == 1].max()
        if (self._k is not None) and (self._k < X.shape[1]):
            # Count the number of false positives (negative instances in box) in each feature's
            # interval and reset the bounds for all features not in the bottom-k regardings FPs
            n_feature_fps = ((X[y == 0] >= self._box_lbs) & (X[y == 0] <= self._box_ubs)).sum()
            exclude_features = n_feature_fps.sort_values().index[self._k:]  # n-k highest
            self._box_lbs[exclude_features] = float('-inf')
            self._box_ubs[exclude_features] = float('inf')
        end_time = time.process_time()
        # Post-processing (as for optimizer-based solutions): if box extends to the limit of
        # feature values in the given data, treat this value as unbounded
        self._box_lbs[self._box_lbs == X.min()] = float('-inf')
        self._box_ubs[self._box_ubs == X.max()] = float('inf')
        return {'objective_value': None,  # methods picks bounds without computing its objective
                'optimization_status': None,
                'optimization_time': end_time - start_time}


class RandomSubgroupDiscoverer(SubgroupDiscoverer):
    """Random-sampling baseline for subgroup discovery

    Choose bound candidates repeatedly uniformly random from the unique values of each feature and
    return the sampled bounds with the highest WRAcc after a fixed number of iterations.
    """

    # Initialize fields.
    # - "k" is the maximum number of features that may be selected in the subgroup description.
    # - "n_iters" is the number of repetitions of random sampling.
    def __init__(self, k: Optional[int] = None, n_iters: int = 1000):
        super().__init__()
        self._k = k
        self._n_iters = n_iters

    # Repeatedly sample random boxes and pick the best of them after a fixed number of iterations.
    # Return meta-data about the fitting process (see superclass for more details).
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        assert y.isin((0, 1, False, True)).all(), 'Target "y" needs to be binary (bool or int).'
        unique_values_per_feature = [sorted(X[col].unique().tolist()) for col in X.columns]
        X_np = X.values  # working directly on numpy arrays rather than pandas faster
        y_np = y.values
        rng = random.Random(25)
        # "Optimization": Repeated random sampling
        start_time = time.process_time()
        opt_quality = float('-inf')
        for _ in range(self._n_iters):
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
        return {'objective_value': opt_quality,
                'optimization_status': None,
                'optimization_time': end_time - start_time}


class PRIMSubgroupDiscoverer(SubgroupDiscoverer):
    """PRIM algorithm

    Heuristic search procedure with a peeling phase (iteratively decreasing the bounds of the
    subgroup) and a pasting phase (iteratively increasing the bounds of the subgroup). Here, we
    only implement the peeling phase. It iteratively sets bounds (on one feature per iteration)
    such that a fraction "alpha" of instances gets removed from the box. It continues till only
    a certain fraction "beta_0" of instances remains in the box and returns the box with the
    highest WRAcc from all iterations.

    Our implementation is similar to the PRIM implementation in prelim.sd.PRIM, but

    - has a different termination condition (min support, as in the original paper, instead of
      a fixed iteration count combined with an early-termination criterion)
    - handles bounds differently (always produces strict bounds first, as in the original paper,
      but converts to <=/>= later, for compatibility with our other subgroup-discovery methods)
    - supports a cardinality constraint on the number of restricted features

    Literature:
    - Friedman & Fisher (1999): "Bump hunting in high-dimensional data"
    - Arzamasov et al. (2021): "REDS: Rule Extraction for Discovering Scenarios"
    - https://github.com/Arzik1987/prelim/blob/main/src/prelim/sd/PRIM.py
    """

    # Initialize fields.
    # - "k" is the maximum number of features that may be selected in the subgroup description.
    # - "alpha" is the fraction of instances peeled off per iteration.
    # - "beta_0" is the minimum fraction of instances in the box to continue peeling.
    def __init__(self, k: Optional[int] = None, alpha: float = 0.05, beta_0: float = 0):
        super().__init__()
        self._k = k
        self._alpha = alpha
        self._beta_0 = beta_0

    # Iteratively peel off (at least) a fraction "_alpha" of the instances by moving one bound of
    # one feature (choose best WRAcc); stop if support threshold "_beta_0" violated (by default, if
    # empty box produced, but can be set higher). Return meta-data about the fitting process (see
    # superclass for more details).
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        assert y.isin((0, 1, False, True)).all(), 'Target "y" needs to be binary (bool or int).'
        X_np = X.values  # working directly on numpy arrays rather than pandas sometimes way faster
        y_np = y.values
        # Optimization: Iterative box updates
        start_time = time.process_time()
        self._box_lbs = np.repeat(float('-inf'), repeats=X_np.shape[1])
        self._box_ubs = np.repeat(float('inf'), repeats=X_np.shape[1])
        y_pred = self.predict_np(X=X_np)
        opt_quality = wracc_np(y_true=y_np, y_pred=y_pred)
        opt_box_lbs = self._box_lbs.copy()  # fields will be changed for predictions, so copy
        opt_box_ubs = self._box_ubs.copy()
        # Peeling continues as long as box contains certain number of instances:
        while np.count_nonzero(y_pred) / len(y_np) > self._beta_0:
            # Note that peeling also changes values of "self._box_lbs" and "self._box_ubs"
            # (i.e., these fields are updated once each iteration and represent current box)
            self._peel_one_step(X=X_np, y=y_np)
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
        return {'objective_value': opt_quality,
                'optimization_status': None,
                'optimization_time': end_time - start_time}

    # Determine which features may be selected for refining the current subgroup defined by bounds
    # on "X", considering a feature-cardinality constraint and removing constant features. Return
    # the (column) indices of permissible features.
    def _get_permissible_feature_idxs(self, X: np.array) -> Sequence[float]:
        permissible_feature_idxs = range(X.shape[1])
        if self._k is not None:
            is_feature_selected = ((X < self._box_lbs) | (X > self._box_ubs)).any(axis=0)
            selected_feature_idxs = np.where(is_feature_selected)[0]
            if len(selected_feature_idxs) == self._k:
                permissible_feature_idxs = selected_feature_idxs
            elif len(selected_feature_idxs) > self._k:
                raise RuntimeError('The algorithm selected more features than allowed.')
        # Exclude features only having one unique value (i.e., all feature values equal first one):
        permissible_feature_idxs = [j for j in permissible_feature_idxs if (X[:, j] != X[0, j]).any()]
        return permissible_feature_idxs

    # For each feature, check the "alpha" / "1 -  alpha" quantile for instances in the box as
    # potential new lower / upper bound. Choose the feature and bound with the best objective
    # (WRAcc) value. If only empty boxes are produced or an empty box is optimal, return it
    # (which causes main algorithm to stop; the latter will not return the empty box, however,
    # as empty box has same WRAcc as unrestricted box, which is used in initialization).
    def _peel_one_step(self, X: np.ndarray, y: np.ndarray) -> bool:
        is_instance_in_old_box = self.predict_np(X=X)
        opt_quality = float('-inf')  # select one peel even if it's not better than previous box
        opt_feature_idx = None
        opt_bound = None
        opt_is_ub = None  # either LB or UB updated
        # Ensure feature-cardinality constraint and exclude features only having one unique value:
        permissible_feature_idxs = self._get_permissible_feature_idxs(X=X)
        if len(permissible_feature_idxs) == 0:  # no peeling possible
            return  # leave method (returned value is not processed, so we return nothing)
        for j in permissible_feature_idxs:
            # Check a new lower bound (if quantile between two feature values, choose middle):
            bound = np.quantile(X[is_instance_in_old_box, j], q=self._alpha, method='midpoint')
            # Only checking the new bound and combining with prior information (on whether instance
            # in box) is faster than updating "self._box_lbs" and using predict_np();
            # also, we use strict equality (as in original paper) here, but will make it >= later
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
            # Convert "<", potentially for a midpoint value, to "<=" for an actual feature value:
            in_box_values = X[is_instance_in_old_box & (X[:, opt_feature_idx] < opt_bound),
                              opt_feature_idx]
            if len(in_box_values) > 0:
                self._box_ubs[opt_feature_idx] = float(in_box_values.max())
            else:  # produce an empty (and invalid) box, causing main algorithm to stop
                self._box_lbs[opt_feature_idx] = float('inf')
                self._box_ubs[opt_feature_idx] = float('-inf')
        else:
            in_box_values = X[is_instance_in_old_box & (X[:, opt_feature_idx] > opt_bound),
                              opt_feature_idx]
            if len(in_box_values) > 0:
                self._box_lbs[opt_feature_idx] = float(in_box_values.min())
            else:  # produce an empty (and invalid) box, causing main algorithm to stop
                self._box_lbs[opt_feature_idx] = float('inf')
                self._box_ubs[opt_feature_idx] = float('-inf')


class BeamSearchSubgroupDiscoverer(AlternativeSubgroupDiscoverer):
    """Beam-search algorithm

    Heuristic search procedure that maintains a beam (list) of candidate boxes and iteratively
    refines them, each iteration testing all possible changes of lower and upper bounds per
    candidate box (but only one change at a time) and retaining a certain (beam width) number of
    boxes with the highest quality. Finally, returns the optimal box from the beam once the beam
    does not change anymore.

    Inspired by the beam-search implementation in pysubgroup.BeamSearch, but faster and supports
    a cardinality constraint on the number of restricted features as well as searching alternative
    subgroup descriptions.

    Literature: https://github.com/flemmerich/pysubgroup/blob/master/src/pysubgroup/algorithms.py
    """

    # Initialize fields.
    # - "k" is the maximum number of features that may be selected in the subgroup description.
    # - "beam_width" is the number of candidate subgroups kept per iteration (lower == faster but
    #   potentially lower quality).
    # - "a" is the number of alternative subgroup descriptions; if None, then none (only original
    #   subgroup searched). Should only be set if "k" is set (else there may be no alternatives).
    # - "tau_abs" is the number of features selected in each existing subgroup description that
    #   should be deselected (*not* be selected) in alternative subgroup description; parameter
    #   should be set if and only if "a" is set.
    def __init__(self, k: Optional[int] = None, beam_width: int = 10,
                 a: Optional[int] = None, tau_abs: Optional[int] = None):
        super().__init__(a=a, tau_abs=tau_abs)
        self._k = k
        self._beam_width = beam_width

    # Determine which features may be selected for refining the current subgroup defined by
    # "bounds" (which is a matrix with two rows (LBs/UBs) and one column per feature) on "X_np",
    # considering (1) a feature-cardinality constraint and (2) constraints for alternatives (if
    # "was_feature_selected_np" is passed, which is a matrix where rows denote existing subgroups
    # and columns denote features). Return the (column) indices of permissible features.
    def _get_permissible_feature_idxs(self, X_np: np.array, bounds: np.array,
                                      was_feature_selected_np: Optional[np.array] = None
                                      ) -> Sequence[float]:
        is_alternative = was_feature_selected_np is not None
        permissible_feature_idxs = range(X_np.shape[1])
        if (self._k is not None) or is_alternative:
            is_feature_selected = ((X_np < bounds[0]) | (X_np > bounds[1])).any(axis=0)
            selected_feature_idxs = np.where(is_feature_selected)[0]
        if self._k is not None:
            if len(selected_feature_idxs) == self._k:
                permissible_feature_idxs = selected_feature_idxs
            elif len(selected_feature_idxs) > self._k:
                raise RuntimeError('The algorithm selected more features than allowed.')
        if is_alternative:
            new_permissible_feature_idxs = []
            # For each existing subgroup, count features not selected in current subgroup
            # (as "was_feature_selected_np" is matrix with one row per existing subgroup, compute
            # row-wise AND and then sum over columns aka features):
            deselection_counts = np.count_nonzero(was_feature_selected_np & ~is_feature_selected,
                                                  axis=1)
            # Minimum dissimilarity regarding each existing subgroup depends on number of actually
            # selected features (latter may be < "k", so "tau_abs" decreased to avoid infeasiblity)
            tau_abs_adapted = np.minimum(self._tau_abs, was_feature_selected_np.sum(axis=1))
            for j in permissible_feature_idxs:
                # If candidate feature "j" freshly enters the subgroup (was not selected in current
                # subgroup before) but was selected in other subgroup, deselection count decreases
                # by one, else (was already selected in current subgroup or was not selected in
                # other subgroup) deselection count remains the same:
                if is_feature_selected[j]:
                    deselection_counts_j = deselection_counts
                else:
                    deselection_counts_j = deselection_counts - was_feature_selected_np[:, j]
                if (deselection_counts_j >= tau_abs_adapted).all():
                    new_permissible_feature_idxs.append(j)
            permissible_feature_idxs = new_permissible_feature_idxs
        return permissible_feature_idxs

    # Iteratively refine boxes; stop if no box in the beam has changed in the previous iteration.
    # Either find original subgroup (if only "X" and "y" are passed) or an alternative subgroup
    # description (if optional arguments are passed). Return meta-data about the optimization
    # process (see superclass for details).
    def _optimize(self, X: pd.DataFrame, y: pd.Series,
                  was_feature_selected_list: Optional[Sequence[Sequence[bool]]] = None,
                  was_instance_in_box: Optional[Sequence[bool]] = None) -> Dict[str, Any]:
        assert y.isin((0, 1, False, True)).all(), 'Target "y" needs to be binary (bool or int).'
        if was_instance_in_box is not None:  # search alternative subgroup description
            objective_func = hamming_np
            y_np = np.array(was_instance_in_box)  # reproduce subgroup, true "y" doesn't matter
            was_feature_selected_np = np.array(was_feature_selected_list)
        else:  # search original subgroup description
            objective_func = wracc_np
            y_np = y.values
            was_feature_selected_np = None
        X_np = X.values  # working directly on numpy arrays rather than pandas faster
        start_time = time.process_time()
        # Initially, all boxes in the beam are unbounded, i.e., limits are (-inf, inf) per feature:
        beam_bounds = np.array([[np.repeat(-np.inf, repeats=X_np.shape[1]),
                                 np.repeat(np.inf, repeats=X_np.shape[1])]
                                for _ in range(self._beam_width)])  # beam width * 2 * num features
        # Initial boxes contain all instances:
        beam_is_in_box = np.array([np.ones(shape=X_np.shape[0], dtype=bool)
                                   for _ in range(self._beam_width)])  # beam width * num instances
        # All boxes should be considered for updates:
        cand_has_changed = np.ones(shape=self._beam_width, dtype=bool)
        # Compute objective value for initial boxes:
        cand_quality = [objective_func(y_np, is_in_box) for is_in_box in beam_is_in_box]
        cand_min_quality = cand_quality.min()
        while np.count_nonzero(cand_has_changed) > 0:  # at least one box changed last iteration
            # Copy boxes from the beam since the candidates for next beam will be updated, but we
            # still want to iterate over the unchanged beam of the previous iteration:
            cand_bounds = beam_bounds.copy()
            cand_is_in_box = beam_is_in_box.copy()
            prev_cand_changed_idxs = np.where(cand_has_changed)[0]
            cand_has_changed = np.zeros(shape=self._beam_width, dtype=bool)
            # Iterate over all previously updated boxes in the beam and try to refine them:
            for box_idx in prev_cand_changed_idxs:
                bounds = beam_bounds[box_idx]
                is_in_box = beam_is_in_box[box_idx]
                # Enforce constrains for feature cardinality and alternatives:
                permissible_feature_idxs = self._get_permissible_feature_idxs(
                    X_np=X_np, bounds=bounds, was_feature_selected_np=was_feature_selected_np)
                for j in permissible_feature_idxs:
                    bound_values = np.unique(X_np[is_in_box, j])  # sorted by default
                    # Test new values for lower bound (> box's previous one but <= box's UB)
                    for bound_value in bound_values[1:]:
                        y_pred = is_in_box & (X_np[:, j] >= bound_value)
                        quality = objective_func(y_np, y_pred)
                        # Replace the minimum-quality candidate, but only if newly created box not
                        # already a candidate (same new box may be created from multiple old boxes)
                        if quality > cand_min_quality:
                            new_bounds = bounds.copy()
                            new_bounds[0, j] = bound_value
                            if all((x != new_bounds).any() for x in cand_bounds):
                                min_quality_idx = np.where(cand_quality == cand_min_quality)[0][0]
                                cand_bounds[min_quality_idx] = new_bounds
                                cand_is_in_box[min_quality_idx] = y_pred
                                cand_has_changed[min_quality_idx] = True
                                cand_quality[min_quality_idx] = quality
                                cand_min_quality = cand_quality.min()
                    # Test new values for upper bound (< box's previous one but >= box's LB)
                    for bound_value in bound_values[:-1]:
                        y_pred = is_in_box & (X_np[:, j] <= bound_value)
                        quality = objective_func(y_np, y_pred)
                        if quality > cand_min_quality:
                            new_bounds = bounds.copy()
                            new_bounds[1, j] = bound_value
                            if all((x != new_bounds).any() for x in cand_bounds):
                                min_quality_idx = np.where(cand_quality == cand_min_quality)[0][0]
                                cand_bounds[min_quality_idx] = new_bounds
                                cand_is_in_box[min_quality_idx] = y_pred
                                cand_has_changed[min_quality_idx] = True
                                cand_quality[min_quality_idx] = quality
                                cand_min_quality = cand_quality.min()
            # Best candidates are next beam; copy() unnecessary, as next candidates will be copied:
            beam_bounds = cand_bounds
            beam_is_in_box = cand_is_in_box
        # Select best subgroup out of beam:
        opt_quality = cand_quality.max()
        opt_quality_idx = np.where(cand_quality == opt_quality)[0][0]
        end_time = time.process_time()
        # Post-processing (as for optimizer-based solutions): if box extends to the limit of
        # feature values in the given data, treat this value as unbounded
        self._box_lbs = pd.Series(beam_bounds[opt_quality_idx, 0], index=X.columns)
        self._box_ubs = pd.Series(beam_bounds[opt_quality_idx, 1], index=X.columns)
        self._box_lbs[self._box_lbs == X.min()] = float('-inf')
        self._box_ubs[self._box_ubs == X.max()] = float('inf')
        return {'objective_value': opt_quality,
                'optimization_status': None,
                'optimization_time': end_time - start_time}


class BestIntervalSubgroupDiscoverer(SubgroupDiscoverer):
    """Best-interval algorithm, wrapped in a beam seach

    Heuristic search procedure that maintains a beam (list) of candidate boxes and iteratively
    refines them. The best-interval technique for refinement checks all lower/upper bound
    combinations for each feature but only requires linear instead of quadratic cost regarding
    the number of unique feature values, thanks to WRAcc being an additive metric. For comparison,
    our other beam-search implementation achieves linear cost by only changing either lower or
    upper bound, but not both simultaneously per feature and iteration. Otherwise, the search
    procedures are quite similar.

    A similar implementation can also be found in prelim.sd.BI (seems to be slower on average and
    has an additional termination condition in the form of a hard-coded iteration count).

    Literature:
    - Mampaey et al. (2012): "Efficient Algorithms for Finding Richer Subgroup Descriptions in
      Numeric and Nominal Data"
    - Arzamasov et al. (2021): "REDS: Rule Extraction for Discovering Scenarios"
    - https://github.com/Arzik1987/prelim/blob/main/src/prelim/sd/BI.py
    """

    # Initialize fields.
    # - "k" is the maximum number of features that may be selected in the subgroup description.
    # - "beam_width" is the number of candidate subgroups kept per iteration (lower == faster but
    #   potentially lower quality).
    def __init__(self, k: Optional[int] = None, beam_width: int = 10):
        super().__init__()
        self._k = k
        self._beam_width = beam_width

    # Determine which features may be selected for refining the current subgroup defined by
    # "bounds" (which is a matrix with two rows (LBs/UBs) and one column per feature) on "X_np",
    # considering a feature-cardinality constraint. Return the (column) indices of permissible
    # features.
    def _get_permissible_feature_idxs(self, X_np: np.array, bounds: np.array) -> Sequence[float]:
        permissible_feature_idxs = range(X_np.shape[1])
        if self._k is not None:
            is_feature_selected = ((X_np < bounds[0]) | (X_np > bounds[1])).any(axis=0)
            selected_feature_idxs = np.where(is_feature_selected)[0]
            if len(selected_feature_idxs) == self._k:
                permissible_feature_idxs = selected_feature_idxs
            elif len(selected_feature_idxs) > self._k:
                raise RuntimeError('The algorithm selected more features than allowed.')
        return permissible_feature_idxs

    # Iteratively refine boxes; stop if no box in the beam has changed in the previous iteration.
    # Return meta-data about the fitting process (see superclass for more details).
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        assert y.isin((0, 1, False, True)).all(), 'Target "y" needs to be binary (bool or int).'
        X_np = X.values  # working directly on numpy arrays rather than pandas faster
        y_np = y.values
        start_time = time.process_time()
        # Initially, all boxes in the beam are unbounded, i.e., limits are (-inf, inf) per feature:
        beam_bounds = np.array([[np.repeat(-np.inf, repeats=X_np.shape[1]),
                                 np.repeat(np.inf, repeats=X_np.shape[1])]
                                for _ in range(self._beam_width)])  # beam width * 2 * num features
        # Initial boxes contain all instances:
        beam_is_in_box = np.array([np.ones(shape=X_np.shape[0], dtype=bool)
                                   for _ in range(self._beam_width)])  # beam width * num instances
        # All boxes should be considered for updates:
        cand_has_changed = np.ones(shape=self._beam_width, dtype=bool)
        # Compute objective value for initial boxes:
        beam_quality = [wracc_np(y_true=y_np, y_pred=is_in_box) for is_in_box in beam_is_in_box]
        cand_min_quality = beam_quality.min()
        while np.count_nonzero(cand_has_changed) > 0:  # at least one box changed last iteration
            # Copy boxes from the beam since the candidates for next beam will be updated, but we
            # still want to iterate over the unchanged beam of the previous iteration:
            cand_bounds = beam_bounds.copy()
            cand_is_in_box = beam_is_in_box.copy()
            cand_quality = beam_quality.copy()
            prev_cand_changed_idxs = np.where(cand_has_changed)[0]
            cand_has_changed = np.zeros(shape=self._beam_width, dtype=bool)
            # Iterate over all previously updated boxes in the beam and try to refine them:
            for box_idx in prev_cand_changed_idxs:
                bounds = beam_bounds[box_idx]
                is_in_box = beam_is_in_box[box_idx]
                box_quality = beam_quality[box_idx]
                # If feature-cardinality constraint used, check how many features restricted in
                # box; if already k, only consider these features; else, consider all features:
                permissible_feature_idxs = self._get_permissible_feature_idxs(
                    X_np=X_np, bounds=bounds)
                for j in permissible_feature_idxs:
                    # BestInterval routine for one feature (see Mampaey et al. (2012)):
                    feat_lb_value = bounds[0, j]  # "l" in Mampaey et al. (2012)
                    feat_ub_value = bounds[1, j]  # "r" in Mampaey et al. (2012)
                    feat_is_in_box = is_in_box
                    feat_quality = box_quality  # "WRAcc_{max}" in Mampaey et al. (2012)
                    feat_lb_current_value = None  # "t_{max}" in Mampaey et al. (2012)
                    feat_lb_current_quality = float('-inf')  # "h_{max}" in Mampaey et al. (2012)
                    # Test new values for bounds; leverage that WRAcc is additive to reduce number
                    # of (LB, UB) combinations from quadratic to linear:
                    bound_values = np.unique(X_np[is_in_box, j])  # sorted by default
                    for bound_value in bound_values:
                        y_pred = is_in_box & (X_np[:, j] >= bound_value)
                        quality = wracc_np(y_true=y_np, y_pred=y_pred)
                        if quality > feat_lb_current_quality:
                            feat_lb_current_value = bound_value
                            feat_lb_current_quality = quality
                        y_pred = (is_in_box & (X_np[:, j] >= feat_lb_current_value) &
                                  (X_np[:, j] <= bound_value))
                        quality = wracc_np(y_true=y_np, y_pred=y_pred)
                        if quality > feat_quality:
                            feat_lb_value = feat_lb_current_value
                            feat_ub_value = bound_value
                            feat_is_in_box = y_pred
                            feat_quality = quality
                    # Update list of candidate boxes with best-interval bounds for current feature
                    # in case they are better than one other candidate box and not a duplicate:
                    if feat_quality > cand_min_quality:
                        new_bounds = bounds.copy()
                        new_bounds[0, j] = feat_lb_value
                        new_bounds[1, j] = feat_ub_value
                        if all((x != new_bounds).any() for x in cand_bounds):
                            min_quality_idx = np.where(cand_quality == cand_min_quality)[0][0]
                            cand_bounds[min_quality_idx] = new_bounds
                            cand_is_in_box[min_quality_idx] = feat_is_in_box
                            cand_has_changed[min_quality_idx] = True
                            cand_quality[min_quality_idx] = feat_quality
                            cand_min_quality = cand_quality.min()
            # Best candidates are next beam; copy() unnecessary, as next candidates will be copied:
            beam_bounds = cand_bounds
            beam_is_in_box = cand_is_in_box
            beam_quality = cand_quality
        # Select best subgroup out of beam:
        opt_quality = cand_quality.max()
        opt_quality_idx = np.where(cand_quality == opt_quality)[0][0]
        end_time = time.process_time()
        # Post-processing (as for optimizer-based solutions): if box extends to the limit of
        # feature values in the given data, treat this value as unbounded
        self._box_lbs = pd.Series(beam_bounds[opt_quality_idx, 0], index=X.columns)
        self._box_ubs = pd.Series(beam_bounds[opt_quality_idx, 1], index=X.columns)
        self._box_lbs[self._box_lbs == X.min()] = float('-inf')
        self._box_ubs[self._box_ubs == X.max()] = float('inf')
        return {'objective_value': opt_quality,
                'optimization_status': None,
                'optimization_time': end_time - start_time}
