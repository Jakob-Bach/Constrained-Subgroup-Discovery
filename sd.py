"""Subgroup discovery

Classes for subgroup-discovery methods.
"""


from abc import ABCMeta, abstractmethod
import random
import time
from typing import Any, Dict, Type

from ortools.linear_solver import pywraplp
import pandas as pd
import prelim.sd.BI
import prelim.sd.PRIM
import pysubgroup
import prim
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

        # Optimize and store/return results:
        start_time = time.process_time()
        optimization_status = model.Solve()
        end_time = time.process_time()
        # Directly extracting values of "lb_vars" and "ub_vars" can lead to numeric issues (e.g.,
        # instance from within box might fall slightly outside, even if all numerical tolerances
        # set to 0 in optimization), so we use actual feature values of instances in box instead;
        # nice side-effect: box is tight around instances instead extending into margin around them
        is_instance_in_box = [bool(var.solution_value()) for var in is_instance_in_box_vars]
        self._box_lbs = X.iloc[is_instance_in_box].min()
        self._box_lbs[self._box_lbs == feature_minima] = float('-inf')
        self._box_ubs = X.iloc[is_instance_in_box].max()
        self._box_ubs[self._box_ubs == feature_maxima] = float('inf')
        return {'optimization_status': optimization_status,
                'optimization_time': end_time - start_time}


class SMTSubgroupDiscoverer(SubgroupDiscoverer):
    """SMT-based subgroup-discovery method

    White-box formulation of subgroup discovery as a Satisfiability Modulo Theories (SMT)
    optimization problem.
    """

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

        # Optimize and store/return results:
        start_time = time.process_time()
        optimization_status = optimizer.check()
        end_time = time.process_time()
        # To avoid potential numeric issues when extracting values of real variables, use actual
        # feature values of instances in box as bounds (also makes box tight around instances)
        is_instance_in_box = [bool(optimizer.model()[var]) for var in is_instance_in_box_vars]
        self._box_lbs = X.iloc[is_instance_in_box].min()
        self._box_lbs[self._box_lbs == feature_minima] = float('-inf')
        self._box_ubs = X.iloc[is_instance_in_box].max()
        self._box_ubs[self._box_ubs == feature_maxima] = float('inf')
        return {'optimization_status': str(optimization_status),
                'optimization_time': end_time - start_time}


class MORBSubgroupDiscoverer(SubgroupDiscoverer):
    """MORB (Minimal Optimal-Recall Box) baseline for subgroup discovery

    Choose the bounds as the minimum and maximum feature value of positive instances, so the box
    contains all positive instances and has the minimal size of all boxes doing so. Finds the
    optimal solution if a box exists that contains all positive and no negative instances.
    """

    # Choose the minimal box that still has optimal recall (= 1). Return meta-data about the
    # fitting process (see superclass for more details).
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        assert y.isin((0, 1, False, True)).all(), 'Target "y" needs to be binary (bool or int).'
        # "Optimization": Find minima and maxima of positive instances:
        start_time = time.process_time()
        self._box_lbs = X[y == 1].min()
        self._box_ubs = X[y == 1].max()
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

    # Initialize fields used in the search.
    def __init__(self, n_repetitions: int = 1000):
        super().__init__()
        self._n_repetitions = n_repetitions

    # Repeatedly sample random boxes and pick the best of them. Return meta-data about the fitting
    # process (see superclass for more details).
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        assert y.isin((0, 1, False, True)).all(), 'Target "y" needs to be binary (bool or int).'
        unique_feature_values = [X[col].unique().tolist() for col in X.columns]
        rng = random.Random(25)
        # "Optimization": Repeated random sampling
        start_time = time.process_time()
        opt_quality = float('-inf')
        for _ in range(self._n_repetitions):
            bounds = [rng.sample(x, k=2) for x in unique_feature_values]  # still LB/UB unordered
            self._box_lbs = pd.Series([min(x) for x in bounds], index=X.columns)
            self._box_ubs = pd.Series([max(x) for x in bounds], index=X.columns)
            quality = wracc(y_true=y, y_pred=self.predict(X=X))
            if quality > opt_quality:
                opt_quality = quality
                opt_box_lbs = self._box_lbs
                opt_box_ubs = self._box_ubs
        end_time = time.process_time()
        # Post-processing (as for optimizer-based solutions): if box extends to the limit of
        # feature values in the given data, treat this value as unbounded
        self._box_lbs = opt_box_lbs
        self._box_ubs = opt_box_ubs
        self._box_lbs[self._box_lbs == X.min()] = float('-inf')
        self._box_ubs[self._box_ubs == X.max()] = float('inf')
        return {'optimization_status': None,
                'optimization_time': end_time - start_time}


class PrimPRIMSubgroupDiscoverer(SubgroupDiscoverer):
    """PRIM algorithm from the package "prim"

    Heuristic search procedure with a peeling phase (iteratively decreasing the range of the
    subgroup) and a pasting phase (iteratively increasing the range of the subgroup).

    Literature: Friedman & Fisher (1999): "Bump hunting in high-dimensional data"
    """

    # Run the PRIM algorithm with its default hyperparameters. Return meta-data about the fitting
    # process (see superclass for more details).
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        assert y.isin((0, 1, False, True)).all(), 'Target "y" needs to be binary (bool or int).'
        model = prim.Prim(x=X, y=y)
        start_time = time.process_time()
        box = model.find_box()
        end_time = time.process_time()
        # Box only contains bounds for restricted features; thus, we initialize all features first:
        self._box_lbs = pd.Series([-float('inf')] * X.shape[1], index=X.columns)
        self._box_ubs = pd.Series([float('inf')] * X.shape[1], index=X.columns)
        self._box_lbs[box.limits.index] = box.limits['min']
        self._box_ubs[box.limits.index] = box.limits['max']
        return {'optimization_status': None,
                'optimization_time': end_time - start_time}


class PrelimSubgroupDiscoverer(SubgroupDiscoverer, metaclass=ABCMeta):
    """Subgroup-discovery algorithm from the package "prelim"

    Superclass wrapping algorithms from the package "prelim". Subclasses need to set a concrete
    "model_type" (= algorithm) in the initializer.
    """

    # Sets field for internal model type. Cannot really be any type, but the classes in "prelim"
    # don't have a common superclass. The "model" needs to have a fit() method that returns a box
    # with a special format.
    def __init__(self, model_type: Type[Any]):
        super().__init__()
        self._model_type = model_type

    # Run the algorithm from "prelim" with its default hyperparameters. Return meta-data about the
    # fitting process (see superclass for more details).
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        assert y.isin((0, 1, False, True)).all(), 'Target "y" needs to be binary (bool or int).'
        model = self._model_type()
        start_time = time.process_time()
        model.fit(X=X, y=y)
        end_time = time.process_time()
        # Unrestricted features of box are -/+ inf by default, so no separate initalization needed
        self._box_lbs = pd.Series(model.box_[0], index=X.columns)
        self._box_ubs = pd.Series(model.box_[1], index=X.columns)
        return {'optimization_status': None,
                'optimization_time': end_time - start_time}


class PrelimPRIMSubgroupDiscoverer(PrelimSubgroupDiscoverer):
    """PRIM algorithm from the package "prelim"

    Heuristic search procedure with a peeling phase (iteratively decreasing the range of the
    subgroup) and a pasting phase (iteratively increasing the range of the subgroup).
    In this version of the algorithm, only the peeling phase is implemented.

    Literature: Friedman & Fisher (1999): "Bump hunting in high-dimensional data"
    """

    def __init__(self):
        super().__init__(model_type=prelim.sd.PRIM.PRIM)


class BISubgroupDiscoverer(PrelimSubgroupDiscoverer):
    """BI algorithm from the package "prelim"

    Heuristic search procedure using beam search.

    Literature: Mampaey et a. (2012): "Efficient Algorithms for Finding Richer Subgroup
    Descriptions in Numeric and Nominal Data"
    """

    def __init__(self):
        super().__init__(model_type=prelim.sd.BI.BI)


class PysubgroupSubgroupDiscoverer(SubgroupDiscoverer, metaclass=ABCMeta):
    """Subgroup-discovery algorithm from the package "pysubgroup"

    Superclass wrapping algorithms from the package "pysubgroup". Subclasses need to set a concrete
    "model_type" (= algorithm) in the initializer.
    """

    # Sets field for internal model type. Cannot really be any type, but the classes in
    # "pysubgroup" don't have a common superclass, though they share a common interface
    # (particularly an execute() method).
    def __init__(self, model_type: Type[Any]):
        super().__init__()
        self._model_type = model_type

    # Run the algorithm from "pysubgroup" with its default hyperparameters up a depth of
    # 2 * |features|. Return meta-data about the fitting process (see superclass for more details).
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        assert y.isin((0, 1, False, True)).all(), 'Target "y" needs to be binary (bool or int).'
        model = self._model_type()
        data = pd.concat((X, y), axis='columns')
        target = pysubgroup.BinaryTarget(target_attribute='target', target_value=1)
        # Create box bounds at each feature value manually (pysubgroub.create_selectors() bins
        # numeric features, which we don't want):
        search_space = []
        for feature in X.columns:
            for value in X[feature].unique():
                search_space.append(pysubgroup.IntervalSelector(feature, value, float('inf')))
                search_space.append(pysubgroup.IntervalSelector(feature, float('-inf'), value))
        task = pysubgroup.SubgroupDiscoveryTask(
            data=data, target=target, search_space=search_space, result_set_size=1,
            depth=(2 * X.shape[1]),  qf=pysubgroup.WRAccQF())  # depth = max number of conditions
        start_time = time.process_time()
        results = model.execute(task).to_dataframe()
        end_time = time.process_time()
        # Box only contains bounds for restricted features; thus, we initialize all features first:
        self._box_lbs = pd.Series([-float('inf')] * X.shape[1], index=X.columns)
        self._box_ubs = pd.Series([float('inf')] * X.shape[1], index=X.columns)
        for selector in results['subgroup'].iloc[0].selectors:
            # Subgroup description may contain multiple (redundant) lower or upper bounds for same
            # feature, so we use min/max to only keep the tighest ones:
            self._box_lbs[selector.attribute_name] = max(selector.lower_bound,
                                                         self._box_lbs[selector.attribute_name])
            self._box_ubs[selector.attribute_name] = min(selector.upper_bound,
                                                         self._box_ubs[selector.attribute_name])
        # Pysubgroup uses < (right-open intervals) instead of <= constraints for upper bounds,
        # which we transform to <= regarding next smaller feature value (consistent with other
        # approaches in this file; alternatively, could subtract a small constant)
        for feature in X.columns:
            if self._box_ubs[feature] < float('inf'):
                self._box_ubs[feature] = max(X[feature][X[feature] < self._box_ubs[feature]])
        return {'optimization_status': None,
                'optimization_time': end_time - start_time}


class AprioriSubgroupDiscoverer(PysubgroupSubgroupDiscoverer):
    """Apriori algorithm from the package "pysubgroup"

    Exact search procedure that iteratively forms subgroups by adding (feature-value)
    conditions to the subgroup and uses quality-based pruning.
    """

    def __init__(self):
        super().__init__(model_type=pysubgroup.Apriori)


class BeamSearchSubgroupDiscoverer(PysubgroupSubgroupDiscoverer):
    """Beam-search algorithm from the package "pysubgroup"

    Heuristic search procedure that iteratively forms subgroups by adding (feature-value)
    conditions to the subgroup.
    """

    def __init__(self):
        super().__init__(model_type=pysubgroup.BeamSearch)


class GpGrowthSubgroupDiscoverer(PysubgroupSubgroupDiscoverer):
    """GP-Growth algorithm from the package "pysubgroup"

    Exact search procedure based on frequent pattern growth.
    """

    def __init__(self):
        super().__init__(model_type=pysubgroup.GpGrowth)


class DFSSubgroupDiscoverer(PysubgroupSubgroupDiscoverer):
    """Depth-first-search algorithm from the package "pysubgroup"

    Exact search procedure that recursively adds (feature-value) conditions to the subgroup.
    """

    def __init__(self):
        super().__init__(model_type=pysubgroup.SimpleDFS)
