"""Subgroup-discovery methods

Classes for subgroup-discovery methods: solver-based, heuristics, and baselines. All methods can
search for original subgroups (subclasses of :class:`SubgroupDiscoverer`), some can also search for
alternative subgroup descriptions (subclasses of :class:`AlternativeSubgroupDiscoverer`).

Literature
----------
Bach (2024): "Using Constraints to Discover Sparse and Alternative Subgroup Descriptions"
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

    The abstract base class for subgroup-discovery methods, having a similar interface as
    prediction models in `scikit-learn`. In particular, defines an abstract method for fitting
    (= search for subgroup) named :meth:`fit`, which needs to be overridden in subclasses, and a
    prediction method named `:meth:predict`, which can be used as-is (if fitting is implemented
    properly). The initializer should set hyperparameters of the subgroup-discovery method (so they
    don't need to the passed for fitting). Already implemented functionality (need not be
    overridden) include access to the subgroup's bounds, access to the feature selection, and
    evaluating the method on a train-test split of a dataset.

    Literature
    ----------
    Atzmueller (2015): "Subgroup disocvery"
    """

    def __init__(self):
        """Initialize subgroup-discovery method

        Defines the fields for the subgroup description's lower and upper bounds without setting
        them to valid values (this should happen during fitting). No hyperparameters for subgroup
        discovery set here (since abstract base class), but subclasses should override (and call)
        this method to initialize hyperparameters for fitting. We recommend to at least introduce a
        parameter `k` and set a corresponding field `_k`, which represents a feature-cardinality
        threshold.
        """

        self._box_lbs = None
        self._box_ubs = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Run subgroup-discovery method

        Should run the subgroup-discovery method on the passed data, optimizing a subgroup-quality
        function like Weighted Relative Accuracy (WRAcc). In particular, should update `self`'s
        internal state appropriately (necessary to enable other method calls like predictions) by
        setting the fields :attr:`_box_lbs` and :attr:`_box_ubs` as :class:`pd.Series` (as many
        entries as `X` has columns, using `X`'s column names as indices). If a feature is
        unrestricted, use -/+ infinity as bounds (necessary to determine feature selection).

        Parameters
        ----------
        X : pd.DataFrame
            Dataset (each row is a data object, each column a feature). All values must be numeric.
        y : pd.Series
            Prediction target. Must be boolean (false, true) or binary integer (0, 1) and have the
            same number of entries as `X` has rows.

        Raises
        ------
        NotImplementedError
            Always raised since abstract method (specific to the subgroup-discovery method).

        Returns
        -------
        Dict[str, Any]
            Meta-data about the fitting process. Should contain the keys `objective_value`,
            `optimization_time`, and `optimization_status`; values may also be `None`.
        """

        raise NotImplementedError('Abstract method.')

    def get_box_lbs(self) -> pd.Series:
        """Get lower bounds

        Gets the subgroup description's lower bounds. Should only be called after fitting.

        Returns
        -------
        pd.Series
            The lower bounds. As many entries as there are features, using the feature names as
            indices. Values are -/+ infinity for unrestricted (= unselected) features.
        """

        return self._box_lbs

    def get_box_ubs(self) -> pd.Series:
        """Get upper bounds

        Gets the subgroup description's upper bounds. Should only be called after fitting.

        Returns
        -------
        pd.Series
            The upper bounds. As many entries as there are features, using the feature names as
            indices. Values are -/+ infinity for unrestricted (= unselected) features.
        """

        return self._box_ubs

    def is_feature_selected(self) -> Sequence[bool]:
        """Get binary feature-selection information

        For each feature, gets whether it is selected (excluded at least one training data object
        from the subgroup) or not, based on the subgroup description's internally stored bound
        values. Should only be called after fitting.

        Returns
        -------
        Sequence[bool]
            The feature-selection status. As many entries as there are features.
        """

        return ((self.get_box_lbs() != float('-inf')) |
                (self.get_box_ubs() != float('inf'))).to_list()

    def get_selected_feature_idxs(self) -> Sequence[int]:
        """Get indices of selected features

        Gets the column indices of features that were selected (excluded at least one training data
        object from the subgroup) in the last training, based on the subgroup description's
        internally stored bound values. Should only be called after fitting.

        Returns
        -------
        Sequence[int]
            The indices of selected features. Empty if no feature selected at all.
        """

        return [j for j, is_selected in enumerate(self.is_feature_selected()) if is_selected]

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict subgroup membership

        Given a dataset, predicts for each data object (rows) whether it is in the subgroup or not,
        based on the subgroup description's internally stored bound values. Should only be called
        after fitting.

        Parameters
        ----------
        X : pd.DataFrame
            Dataset (each row is a data object, each column a feature). All values must be numeric.
            Must have the same features (column names) as the dataset used for fitting. The data
            objects can be different.

        Returns
        -------
        pd.Series
            The binary predictions (1 = subgroup member). As many entries as there are rows in `X`.
        """

        return pd.Series((X.ge(self.get_box_lbs()) & X.le(self.get_box_ubs())).all(
            axis='columns').astype(int), index=X.index)

    def predict_np(self, X: np.ndarray) -> np.ndarray:
        """Predict subgroup membership

        Same functionality as :meth:`predict`, but faster and intended for `numpy` arrays as data
        types (for the passed data `X` as well as the internally stored bound values of the
        subgroup description). This routine may be used during fitting in subgroup-discovery
        methods that call prediction often and whose runtime is therefore significantly affected by
        this subroutine. After search, the bound values should be :class:`pd.Series` and the method
        :meth:`predict` be used.

        Parameters
        ----------
        X : np.ndarray
            Dataset (each row is a data object, each column a feature). All values must be numeric.

        Returns
        -------
        prediction : np.ndarray
            The binary predictions (1 = subgroup member). As many entries as there are rows in `X`.
        """

        prediction = np.ones(X.shape[0], dtype=bool)  # start by assuming each instance in box
        for j in range(X.shape[1]):
            prediction = prediction & (X[:, j] >= self._box_lbs[j]) & (X[:, j] <= self._box_ubs[j])
        return prediction

    def evaluate(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame,
                 y_test: pd.Series) -> pd.DataFrame:
        """Evaluate subgroup-discovery method

        Evaluates the subgroup-discovery method on a train-test split of a dataset. Trains on the
        training set, predicts on training + test, and computes evaluation metrics.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training dataset (each row is a data object, each column a feature). All values must be
            numeric.
        y_train : pd.Series
            Training prediction target. Must be boolean (false, true) or binary integer (0, 1) and
            have the same number of entries as `X_train` has rows.
        X_test : pd.DataFrame
            Test dataset (each row is a data object, each column a feature). All values must be
            numeric. Must have the same features (column names) as the training dataset.
        y_test : pd.Series
            Test prediction target. Must be boolean (false, true) or binary integer (0, 1) and have
            the same number of entries as `X_test` has rows.

        Returns
        -------
        pd.DataFrame
            Evaluation results, with subgroups as rows and evaluation metrics + meta-data of the
            subgroups (like bounds) as columns. By default, there only is one row (subgroup), but
            some subgroup-discovery methods in the literature return multiple subgroups
            simultaneously (would need to override this method appropriately).
        """

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

    The abstract base class for subgroup-discovery methods that cannot only find original
    subgroups but also alternative descriptions for the original one. Implements the (formerly
    abstract) :meth:`fit` method by dispatching to a new abstract method :meth:`_optimize`, which
    should be overridden to (based on the passed parameters) either find an orignal subgroup or
    alternative descriptions. :meth:`evaluate` is also overridden to either

    - perform the evaluation routine from the superclass (fit and evaluate original subgroup) or
    - search for original subgroup + alternative descriptions, dispatching to the generic search
      routine :meth:`search_alternative_descriptions` that calls the new optimization method.
      If :meth:`_optimize` is implemented properly, this search routine should also work in
      subclasses without any adaptations necessary.

    Literature
    ----------
    Bach (2024): "Using Constraints to Discover Sparse and Alternative Subgroup Descriptions"
    """

    def __init__(self, a: Optional[int] = None, tau_abs: Optional[int] = None):
        """Initialize subgroup-discovery method

        Defines the fields for the subgroup description's bounds (inherited from superclass) and
        initializes new fields for user parameters in the search for alternative subgroup
        descriptions. Subclasses should override (and call) this method to initialize further
        hyperparameters for fitting. We recommend to at least introduce a method parameter `k` and
        corresponding field `_k`, which represents a feature-cardinality threshold, so there are
        features left for alternatives after searching the original subgroup.

        Parameters
        ----------
        a : Optional[int], optional
            Number of alternative subgroup descriptions. Should be `None` (default) for searching
            original subgroups (:meth:`evaluate` will dispatch accordingly).
        tau_abs : Optional[int], optional
            Number of features selected in the existing subgroup description that should be
            deselected in alternative descriptions. Should be `None` (default) for searching
            original subgroups (:meth:`evaluate` will dispatch accordingly). Should only be set if
            `a` is set as well.
        """

        super().__init__()
        self._a = a
        self._tau_abs = tau_abs

    @abstractmethod
    def _optimize(self, X: pd.DataFrame, y: pd.Series,
                  was_feature_selected_list: Optional[Sequence[Sequence[bool]]] = None,
                  was_instance_in_box: Optional[Sequence[bool]] = None) -> Dict[str, Any]:
        """Run subgroup-discovery method for original or alternative

        Should either find an original subgroup (if only `X` and `y` are passed) or an alternative
        subgroup description (if the optional arguments are passed as well). In particular, should
        update `self`'s internal state appropriately (necessary to enable other method calls like
        predictions) by setting the fields :attr:`_box_lbs` and :attr:`_box_ubs` as
        :class:`pd.Series` (as many entries as `X` has columns, using `X`'s column names as
        indices). If a feature is unrestricted, use -/+ infinity as bounds (necessary to determine
        feature selection).

        Parameters
        ----------
        X : pd.DataFrame
            Dataset (each row is a data object, each column a feature). All values must be numeric.
        y : pd.Series
            Prediction target. Must be boolean (false, true) or binary integer (0, 1) and have the
            same number of entries as `X` has rows.
        was_feature_selected_list : Optional[Sequence[Sequence[bool]]], optional
            For each existing subgroup (outer sequence), indicate for each feature (inner sequence)
            whether selected or not. An alternative subgroup description should deselect (*not*
            select) at least :attr:`_tau_abs` features from each existing subgroup description.
            Should be `None` (default) for searching original subgroups.
        was_instance_in_box : Optional[Sequence[bool]], optional
            For each data object (= instance), indicate whether member of original subgroup or not.
            An alternative subgroup should try to maximize the similarity to this prediction. Must
            have the same number of entries as `X` has rows. Should be `None` (default) for
            searching original subgroups.

        Raises
        ------
        NotImplementedError
            Always raised since abstract method (specific to the subgroup-discovery method).

        Returns
        -------
        Dict[str, Any]
            Meta-data about the optimization process. Should (like the result of :meth:`fit`)
            contain the keys `objective_value`, `optimization_time`, and `optimization_status`;
            values may also be `None`.
        """

        raise NotImplementedError('Abstract method.')

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Run subgroup-discovery method

        Dispatches to `meth`:_optimize:, which should implement the subgroup-discovery method.

        Parameters
        ----------
        X : pd.DataFrame
            Dataset (each row is a data object, each column a feature). All values must be numeric.
        y : pd.Series
            Prediction target. Must be boolean (false, true) or binary integer (0, 1) and have the
            same number of entries as `X` has rows.

        Returns
        -------
        Dict[str, Any]
            Meta-data about the fitting process. Should contain the keys `objective_value`,
            `optimization_time`, and `optimization_status`; values may also be `None`.
        """

        # Dispatch to another, more general routine (which can also find alternative subgroup
        # descriptions; here, consistent to fit() in other classes, only one subgroup searched):
        return self._optimize(X=X, y=y, was_feature_selected_list=None, was_instance_in_box=None)

    def search_alternative_descriptions(self, X_train: pd.DataFrame, y_train: pd.Series,
                                        X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """Search alternative subgroup descriptions

        Evaluates the search for alternative subgroup descriptions on a train-test split of a
        dataset. Trains on the training set, predicts on training + test, and computes evaluation
        metrics. Each of the :attr:`_a` alternative subgroup descriptions should deselect
        (*not* use) at least :attr:`_tau_abs`features from each existing (previous) description.
        The original subgroup should optimize WRAcc or another measure of subgroup quality, all
        subsequent subgroups should optimize normalized Hamming similarity to the original one or
        another measure of subgroup similarity (in both cases, the actual objective depends on the
        implementation of :meth:`_optimize`, which conducts the search for individual subgroups and
        is called multiple times here).

        Parameters
        ----------
        X_train : pd.DataFrame
            Training dataset (each row is a data object, each column a feature). All values must be
            numeric.
        y_train : pd.Series
            Training prediction target. Must be boolean (false, true) or binary integer (0, 1) and
            have the same number of entries as `X_train` has rows.
        X_test : pd.DataFrame
            Test dataset (each row is a data object, each column a feature). All values must be
            numeric. Must have the same features (column names) as the training dataset.
        y_test : pd.Series
            Test prediction target. Must be boolean (false, true) or binary integer (0, 1) and have
            the same number of entries as `X_test` has rows.

        Returns
        -------
        pd.DataFrame
            Evaluation results, with subgroups as rows and evaluation metrics + meta-data of the
            subgroups (like bounds) as columns. The first row contains the original subgroup, the
            subsequent :attr:`_tau_abs` rows contain alternative subgroup descriptions.
        """

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

    def evaluate(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame,
                 y_test: pd.Series) -> pd.DataFrame:
        """Evaluate subgroup-discovery method for original or alternatives

        Depending on whether the fields :attr:`_a` and :attr:`tau_abs` are set (not `None`), either

        - evaluates the subgroup-discovery method for searching an original subgroup (dispatching
          to :meth:`SubgroupDiscoverer:evaluate`) or
        - evaluates the subgroup-discovery method for searching alternative subgroup descriptions
          (dispatching to :meth:`search_alternative_descriptions`).

        Parameters
        ----------
        X_train : pd.DataFrame
            Training dataset (each row is a data object, each column a feature). All values must be
            numeric.
        y_train : pd.Series
            Training prediction target. Must be boolean (false, true) or binary integer (0, 1) and
            have the same number of entries as `X_train` has rows.
        X_test : pd.DataFrame
            Test dataset (each row is a data object, each column a feature). All values must be
            numeric. Must have the same features (column names) as the training dataset.
        y_test : pd.Series
            Test prediction target. Must be boolean (false, true) or binary integer (0, 1) and have
            the same number of entries as `X_test` has rows.

        Returns
        -------
        pd.DataFrame
            Evaluation results, with subgroups as rows and evaluation metrics + meta-data of the
            subgroups (like bounds) as columns. The first row contains the original subgroup, the
            subsequent :attr:`_tau_abs` rows contain alternative subgroup descriptions (if they
            are searched; otherwise, just one row).
        """

        if (self._a is not None) and (self._tau_abs is not None):
            return self.search_alternative_descriptions(X_train=X_train, y_train=y_train,
                                                        X_test=X_test, y_test=y_test)
        else:
            return super().evaluate(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)


class MIPSubgroupDiscoverer(SubgroupDiscoverer):
    """MIP-solver-based subgroup-discovery method

    Solver-based search for subgroups using a white-box formulation of subgroup discovery as a
    Mixed Integer (Linear) Programming (MIP) optimization problem, which is tackled by the solver
    SCIP via the package :mod:`ortools`.

    Literature
    ----------
    Bach (2024): "Using Constraints to Discover Sparse and Alternative Subgroup Descriptions"
    """

    def __init__(self, k: Optional[int] = None, timeout: Optional[float] = None):
        """Initialize subgroup-discovery method

        Defines the fields for the subgroup description's bounds (inherited from superclass) and
        new fields for parameters of this particular subgroup-discovery method.

        Parameters
        ----------
        k : Optional[int], optional
            Feature-cardinality threshold, i.e., maximum number of features that may be selected in
            the subgroup. By default (`None`), all features may be selected.
        timeout : Optional[float], optional
            Timeout of the solver in seconds (according to our experience, may not be observed).
            By default (`None`), no solver timeout.
        """

        super().__init__()
        self._k = k
        self._timeout = timeout

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Run subgroup-discovery method

        Runs the subgroup-discovery method on the passed data, stores bounds internally, and
        returns meta-data about the fitting process.

        Parameters
        ----------
        X : pd.DataFrame
            Dataset (each row is a data object, each column a feature). All values must be numeric.
        y : pd.Series
            Prediction target. Must be boolean (false, true) or binary integer (0, 1) and have the
            same number of entries as `X` has rows.

        Returns
        -------
        Dict[str, Any]
            Meta-data about the fitting process. Contains the keys `objective_value` (WRAcc),
            `optimization_time`, and `optimization_status`. The `optimization_status` is `0` if
            the optimum was found (-> no timeout), `1` if any (potentially suboptimal) solution was
            found (-> timeout), and `6` if no solution was found yet (-> timeout, `objective_value`
            is `NaN`).
        """

        assert y.isin((0, 1, False, True)).all(), 'Target "y" needs to be binary (bool or int).'

        # Define "speaking" names for certain constants in the optimization problem:
        n_instances = X.shape[0]
        n_features = X.shape[1]
        n_pos_instances = y.sum()
        feature_minima = X.min().to_list()
        feature_maxima = X.max().to_list()
        feature_diff_minima = X.apply(lambda col: pd.Series(col.sort_values().unique()).diff(
            ).min()).fillna(0).to_list()
        # fillna() covers the case that all values are identical, for which we want the strict
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
    """SMT-solver-based subgroup-discovery method

    Solver-based search for subgroups using a white-box formulation of subgroup discovery as a
    Satisfiability Modulo Theories (SMT) optimization problem, which is tackled by the solver `Z3`
    via the package :mod:`z3`.

    Literature
    ----------
    Bach (2024): "Using Constraints to Discover Sparse and Alternative Subgroup Descriptions"
    """

    def __init__(self, k: Optional[int] = None, a: Optional[int] = None,
                 tau_abs: Optional[int] = None, timeout: Optional[float] = None):
        """Initialize subgroup-discovery method

        Defines the fields for the subgroup description's bounds and user parameters in the search
        for alternative subgroup descriptions (inherited from superclass) and new fields for
        parameters of this particular subgroup-discovery method.

        Parameters
        ----------
        k : Optional[int], optional
            Feature-cardinality threshold, i.e., maximum number of features that may be selected in
            the subgroup. By default (`None`), all features may be selected.
        a : Optional[int], optional
            Number of alternative subgroup descriptions. Should be `None` (default) for searching
            original subgroups (:meth:`evaluate` will dispatch accordingly).
        tau_abs : Optional[int], optional
            Number of features selected in existing subgroup description that should be deselected
            in alternative description. Should be `None` (default) for searching original subgroups
            (:meth:`evaluate` will dispatch accordingly). Should only be set if `a` is set as well.
        timeout : Optional[float], optional
            Timeout of the solver in seconds. By default (`None`), no solver timeout.
        """

        super().__init__(a=a, tau_abs=tau_abs)
        self._k = k
        self._timeout = timeout

    def _optimize(self, X: pd.DataFrame, y: pd.Series,
                  was_feature_selected_list: Optional[Sequence[Sequence[bool]]] = None,
                  was_instance_in_box: Optional[Sequence[bool]] = None) -> Dict[str, Any]:
        """Run subgroup-discovery method for original or alternative

        Either finds an original subgroup (if only `X` and `y` are passed) or an alternative
        subgroup description (if the optional arguments are passed as well). Stores bounds
        internally, and returns meta-data about the optimization process.

        Parameters
        ----------
        X : pd.DataFrame
            Dataset (each row is a data object, each column a feature). All values must be numeric.
        y : pd.Series
            Prediction target. Must be boolean (false, true) or binary integer (0, 1) and have the
            same number of entries as `X` has rows.
        was_feature_selected_list : Optional[Sequence[Sequence[bool]]], optional
            For each existing subgroup (outer sequence), indicate for each feature (inner sequence)
            whether selected or not. An alternative subgroup description must deselect (*not*
            select) at least :attr:`_tau_abs` features from each existing subgroup description.
            Should be `None` (default) for searching original subgroups.
        was_instance_in_box : Optional[Sequence[bool]], optional
            For each data object (= instance), indicate whether member of original subgroup or not.
            Must have the same number of entries as `X` has rows. Should be `None` (default) for
            searching original subgroups.

        Returns
        -------
        Dict[str, Any]
            Meta-data about the optimization process. Contains the keys `objective_value`,
            `optimization_time`, and `optimization_status`. The `optimization_status` is `"sat"` if
            the optimum was found (-> no timeout) and `"unknown"` if not (-> no timeout, suboptimal
            solution may still be found). The `objective_value` is WRAcc when searching for
            original subgroups and normalized Hamming similarity when searching for alternative
            subgroup descriptions.
        """

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

    Baseline that chooses the subgroup description's bounds as the minimum and maximum feature
    values of all positive data objects, so the subgroup

    - contains all positive data objects (i.e., has optimal recall) and
    - has the minimal size of all subgroup descriptions doing so, i.e., minimizes the number of
      false positives (or, equivalently, maximizes the number of true negatives).

    Finds a perfect subgroup (containing all positive and no negative data objects) if it exists.

    Literature
    ----------
    Bach (2024): "Using Constraints to Discover Sparse and Alternative Subgroup Descriptions"
    """

    def __init__(self, k: Optional[int] = None):
        """Initialize subgroup-discovery method

        Defines the fields for the subgroup description's bounds (inherited from superclass) and
        new fields for parameters of this particular subgroup-discovery method.

        Parameters
        ----------
        k : Optional[int], optional
            Feature-cardinality threshold, i.e., maximum number of features that may be selected in
            the subgroup. By default (`None`), all features may be selected.
        """

        super().__init__()
        self._k = k

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Run subgroup-discovery method

        Runs the subgroup-discovery method on the passed data, stores bounds internally, and
        returns meta-data about the fitting process.

        Parameters
        ----------
        X : pd.DataFrame
            Dataset (each row is a data object, each column a feature). All values must be numeric.
        y : pd.Series
            Prediction target. Must be boolean (false, true) or binary integer (0, 1) and have the
            same number of entries as `X` has rows.

        Returns
        -------
        Dict[str, Any]
            Meta-data about the fitting process. Contains the keys `objective_value`,
            `optimization_time`, and `optimization_status`. The values of `objective_value` and
            `optimization_status` are `None` since the method always returns its local optimum but
            does not explicitly compute an objective value.
        """

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

    Baseline that chooses bound candidates for subgroup descriptions repeatedly uniformly random
    from the unique values of each feature and returns the sampled bounds with the highest WRAcc
    after a fixed number of iterations. If the dataset has many features and no feature-cardinality
    constraints are employed, high likelihood of creating a very small or even empty an subgroup.

    Literature
    ----------
    Bach (2024): "Using Constraints to Discover Sparse and Alternative Subgroup Descriptions"
    """

    def __init__(self, k: Optional[int] = None, n_iters: int = 1000):
        """Initialize subgroup-discovery method

        Defines the fields for the subgroup description's bounds (inherited from superclass) and
        new fields for parameters of this particular subgroup-discovery method.

        Parameters
        ----------
        k : Optional[int], optional
            Feature-cardinality threshold, i.e., maximum number of features that may be selected in
            the subgroup. By default (`None`), all features may be selected.
        n_iters : int, optional
            Number of repetitions for random sampling. The default is 1000.
        """

        super().__init__()
        self._k = k
        self._n_iters = n_iters

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Run subgroup-discovery method

        Runs the subgroup-discovery method on the passed data, stores bounds internally, and
        returns meta-data about the fitting process.

        Parameters
        ----------
        X : pd.DataFrame
            Dataset (each row is a data object, each column a feature). All values must be numeric.
        y : pd.Series
            Prediction target. Must be boolean (false, true) or binary integer (0, 1) and have the
            same number of entries as `X` has rows.

        Returns
        -------
        Dict[str, Any]
            Meta-data about the fitting process. Contains the keys `objective_value` (WRacc),
            `optimization_time`, and `optimization_status`. The value of `optimization_status` is
            always `None` since the method always returns its local optimum.
        """

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
    """PRIM algorithm for subgroup discovery

    Heuristic search method with a peeling phase (iteratively shrinking the subgroup) and a pasting
    phase (iteratively enlarging the subgroup). Here, we only implement the peeling phase.
    It iteratively sets new bounds (on one feature per iteration) such that a fraction
    :attr:`_alpha` of data objects gets removed from the subgroups. It continues until only a
    fraction :attr:`_beta_0` of data objects remains in the subgroup and returns the subgroup
    description with the highest subgroup quality (WRAcc) from all iterations.

    Our implementation is similar to the PRIM implementation in :class:`prelim.sd.PRIM`, but

    - has a different termination condition (minimum support, as in the original paper, instead of
      a fixed iteration count combined with an early-termination criterion),
    - handles bounds differently (always produces strict bounds first, as in the original paper,
      but converts to non-restrict <=/>= later, for compatibility with the other subgroup-discovery
      methods in this package), and
    - supports a feature-cardinality constraint.

    Literature
    ----------
    - Friedman & Fisher (1999): "Bump hunting in high-dimensional data"
    - Arzamasov & Böhm (2021): "REDS: Rule Extraction for Discovering Scenarios"
    - https://github.com/Arzik1987/prelim/blob/main/src/prelim/sd/PRIM.py
    """

    def __init__(self, k: Optional[int] = None, alpha: float = 0.05, beta_0: float = 0):
        """Initialize subgroup-discovery method

        Defines the fields for the subgroup description's bounds (inherited from superclass) and
        new fields for parameters of this particular subgroup-discovery method.

        Parameters
        ----------
        k : Optional[int], optional
            Feature-cardinality threshold, i.e., maximum number of features that may be selected in
            the subgroup. By default (`None`), all features may be selected.
        alpha : float, optional
            Fraction of data objects that is peeled off (excluded from the subgroup) in each
            iteration, i.e., the "patience" of the method. The default is 0.05.
        beta_0 : float, optional
            Minimum fraction of data objects remaining in the subgroup during peeling. Determines
            when peeling stops. The default is 0, i.e., continue peeling till subgroup empty.
        """

        super().__init__()
        self._k = k
        self._alpha = alpha
        self._beta_0 = beta_0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Run subgroup-discovery method

        Runs the subgroup-discovery method on the passed data, stores bounds internally, and
        returns meta-data about the fitting process.

        Parameters
        ----------
        X : pd.DataFrame
            Dataset (each row is a data object, each column a feature). All values must be numeric.
        y : pd.Series
            Prediction target. Must be boolean (false, true) or binary integer (0, 1) and have the
            same number of entries as `X` has rows.

        Returns
        -------
        Dict[str, Any]
            Meta-data about the fitting process. Contains the keys `objective_value` (WRacc),
            `optimization_time`, and `optimization_status`. The value of `optimization_status` is
            always `None` since the method always returns its local optimum.
        """

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

    def _get_permissible_feature_idxs(self, X: np.ndarray) -> Sequence[int]:
        """Get indices of permissible features

        Subroutine for :meth:`fit`. Determines which features from `X` may be selected for refining
        (setting new bounds or adapting existing bounds) the current subgroup with its bounds
        :attr:`_box_lbs` and :attr:`_box_ubs`. Removes constant features (where PRIM cannot set new
        bounds) and considers a feature-cardinality constraint (if :attr:`_k` is set). At most, all
        features are permissible.

        Parameters
        ----------
        X : np.ndarray
            Dataset (each row is a data object, each column a feature). All values must be numeric.

        Raises
        ------
        RuntimeError
            If the current feature selection is invalid already (i.e., violates constraints).
            Should only happen if our implementation contains an unknown bug, because otherwise the
            sole purpose of this method is creating valid feature sets only.

        Returns
        -------
        Sequence[int]
            The indices of features in `X` that are permissible for refinement.
        """
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

    def _peel_one_step(self, X: np.ndarray, y: np.ndarray) -> None:
        """One peeling step within PRIM

        Subroutine for :meth:`fit`, called in each iteration. For each feature, checks the
        :attr:`_alpha` and  1 -  :attr:`_alpha` quantile value for data objects in the subgroup as
        potential new lower / upper bound. Chooses the feature and bound with the best objective
        value (WRAcc) and sets the corresponding bound directly in :attr:`_box_lbs` or
        :attr:`_box_ubs` (instead of returning it). If only empty subgroups are produced or an
        empty subgroup is optimal, sets corresponding bounds (which causes main algorithm to stop;
        the latter will not return the empty subgroup, however, as the empty subgroup has the same
        WRAcc as an unrestricted subgroup description, which is used in PRIM's initialization).

        Parameters
        ----------
        X : np.ndarray
            Dataset (each row is a data object, each column a feature). All values must be numeric.
        y : np.ndarray
            Prediction target. Must be boolean (false, true) or binary integer (0, 1) and have the
            same number of entries as `X` has rows.
        """

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
    """Beam-search algorithm for subgroup discovery

    Heuristic search method that maintains a beam (list) of candidate subgroups and iteratively
    refines them. Each refinement step iteratively tests all possible changes of lower or upper
    bounds per candidate subgroup (but only one change at a time) and retains a certain (beam
    width) number of subgroups with the highest quality. Returns the optimal subgroup from the beam
    once the beam does not change anymore.

    Inspired by the beam-search implementation in :class:`pysubgroup.BeamSearch`, but

    - faster on average,
    - supports a feature-cardinality constraint, and
    - supports searching alternative subgroup descriptions.

    Literature
    ----------
    https://github.com/flemmerich/pysubgroup/blob/master/src/pysubgroup/algorithms.py
    """

    def __init__(self, k: Optional[int] = None, a: Optional[int] = None,
                 tau_abs: Optional[int] = None, beam_width: int = 10):
        """Initialize subgroup-discovery method

        Defines the fields for the subgroup description's bounds and user parameters in the search
        for alternative subgroup descriptions (inherited from superclass) and new fields for
        parameters of this particular subgroup-discovery method.

        Parameters
        ----------
        k : Optional[int], optional
            Feature-cardinality threshold, i.e., maximum number of features that may be selected in
            the subgroup. By default (`None`), all features may be selected.
        a : Optional[int], optional
            Number of alternative subgroup descriptions. Should be `None` (default) for searching
            original subgroups (:meth:`evaluate` will dispatch accordingly).
        tau_abs : Optional[int], optional
            Number of features selected in existing subgroup description that should be deselected
            in alternative description. Should be `None` (default) for searching original subgroups
            (:meth:`evaluate` will dispatch accordingly). Should only be set if `a` is set as well.
        beam_width : int, optional
            Number of candidate subgroups kept per iteration (lower means faster but potentially
            lower subgroup quality). The default is 10.
        """

        super().__init__(a=a, tau_abs=tau_abs)
        self._k = k
        self._beam_width = beam_width

    def _get_permissible_feature_idxs(self, X_np: np.ndarray, bounds: np.ndarray,
                                      was_feature_selected_np: Optional[np.ndarray] = None
                                      ) -> Sequence[int]:
        """Get indices of permissible features

        Subroutine for :meth:`_optimize`. Determines which features from `X` may be selected for
        refining (setting new bounds or adapting existing bounds) the current subgroup with its
        passed `bounds`. Considers a feature-cardinality constraint (if :attr:`_k` is set) and
        constraints for alternative subgroup descriptions (if `was_feature_selected_np` is passed;
        additionally, :attr:`_tau_abs` should be set). At most, all features are permissible.

        Parameters
        ----------
        X : np.ndarray
            Dataset (each row is a data object, each column a feature). All values must be numeric.
        bounds : np.ndarray
            The bounds of the current subgroup as a two-dimensional array (first row contains lower
            bounds, second row contains upper bounds, columns represent features).
        was_feature_selected_np : Optional[np.ndarray], optional
            Binary feature-selection decisions as a two-dimensional array (rows represent existing
            subgroups, columns represent features). The default (`None`) indicates that constraints
            for alternative subgroup descriptions should not be checked.

        Raises
        ------
        RuntimeError
            If the current feature selection is invalid already (i.e., violates constraints).
            Should only happen if our implementation contains an unknown bug, because otherwise the
            sole purpose of this method is creating valid feature sets only.

        Returns
        -------
        Sequence[int]
            The indices of features in `X_np` that are permissible for refinement.
        """

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

    def _optimize(self, X: pd.DataFrame, y: pd.Series,
                  was_feature_selected_list: Optional[Sequence[Sequence[bool]]] = None,
                  was_instance_in_box: Optional[Sequence[bool]] = None) -> Dict[str, Any]:
        """Run subgroup-discovery method for original or alternative

        Either finds an original subgroup (if only `X` and `y` are passed) or an alternative
        subgroup description (if the optional arguments are passed as well). Stores bounds
        internally, and returns meta-data about the optimization process.

        Parameters
        ----------
        X : pd.DataFrame
            Dataset (each row is a data object, each column a feature). All values must be numeric.
        y : pd.Series
            Prediction target. Must be boolean (false, true) or binary integer (0, 1) and have the
            same number of entries as `X` has rows.
        was_feature_selected_list : Optional[Sequence[Sequence[bool]]], optional
            For each existing subgroup (outer sequence), indicate for each feature (inner sequence)
            whether selected or not. An alternative subgroup description must deselect (*not*
            select) at least :attr:`_tau_abs` features from each existing subgroup description.
            Should be `None` (default) for searching original subgroups.
        was_instance_in_box : Optional[Sequence[bool]], optional
            For each data object (= instance), indicate whether member of original subgroup or not.
            Must have the same number of entries as `X` has rows. Should be `None` (default) for
            searching original subgroups.

        Returns
        -------
        Dict[str, Any]
            Meta-data about the optimization process. Contains the keys `objective_value`,
            `optimization_time`, and `optimization_status`. The value of `optimization_status` is
            always `None` since the method always returns its local optimum. The `objective_value`
            is WRAcc when searching for original subgroups and normalized Hamming similarity when
            searching for alternative subgroup descriptions.
        """

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
        cand_quality = np.array([objective_func(y_np, is_in_box) for is_in_box in beam_is_in_box])
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
    """Best-interval algorithm for subgroup discovery, wrapped in a beam seach

    Heuristic search method that maintains a beam (list) of candidate boxes and iteratively
    refines them. The best-interval technique for refinement checks all lower/upper bound
    combinations for each feature but only requires linear instead of quadratic cost regarding
    the number of unique feature values, thanks to WRAcc being an additive metric. For comparison,
    our other beam-search implementation (:class:`BeamSearchSubgroupDiscoverer`) achieves linear
    cost by only changing either lower or upper bound, but not both simultaneously per feature and
    iteration. Otherwise, the search procedures are quite similar.

    Similar to the implementation in :class:`prelim.sd.BI`, but

    - it seems to be faster on average and
    - does not use a hard-coded iteration count as additional termination condition, and
    - supports a feature-cardinality constraint.

    Literature
    ----------
    - Mampaey et al. (2012): "Efficient Algorithms for Finding Richer Subgroup Descriptions in
      Numeric and Nominal Data"
    - Arzamasov & Böhm (2021): "REDS: Rule Extraction for Discovering Scenarios"
    - https://github.com/Arzik1987/prelim/blob/main/src/prelim/sd/BI.py
    """

    def __init__(self, k: Optional[int] = None, beam_width: int = 10):
        """Initialize subgroup-discovery method

        Defines the fields for the subgroup description's bounds (inherited from superclass) and
        new fields for parameters of this particular subgroup-discovery method.

        Parameters
        ----------
        k : Optional[int], optional
            Feature-cardinality threshold, i.e., maximum number of features that may be selected in
            the subgroup. By default (`None`), all features may be selected.
        beam_width : int, optional
            Number of candidate subgroups kept per iteration (lower means faster but potentially
            lower subgroup quality). The default is 10.
        """

        super().__init__()
        self._k = k
        self._beam_width = beam_width

    def _get_permissible_feature_idxs(self, X_np: np.ndarray, bounds: np.ndarray) -> Sequence[int]:
        """Get indices of permissible features

        Subroutine for :meth:`fit`. Determines which features from `X` may be selected for
        refining (setting new bounds or adapting existing bounds) the current subgroup with its
        passed `bounds`. Considers a feature-cardinality constraint (if :attr:`_k` is set).
        At most, all features are permissible.

        Parameters
        ----------
        X : np.ndarray
            Dataset (each row is a data object, each column a feature). All values must be numeric.
        bounds : np.ndarray
            The bounds of the current subgroup as a two-dimensional array (first row contains lower
            bounds, second row contains upper bounds, columns represent features).

        Raises
        ------
        RuntimeError
            If the current feature selection is invalid already (i.e., violates constraints).
            Should only happen if our implementation contains an unknown bug, because otherwise the
            sole purpose of this method is creating valid feature sets only.

        Returns
        -------
        Sequence[int]
            The indices of features in `X_np` that are permissible for refinement.
        """

        permissible_feature_idxs = range(X_np.shape[1])
        if self._k is not None:
            is_feature_selected = ((X_np < bounds[0]) | (X_np > bounds[1])).any(axis=0)
            selected_feature_idxs = np.where(is_feature_selected)[0]
            if len(selected_feature_idxs) == self._k:
                permissible_feature_idxs = selected_feature_idxs
            elif len(selected_feature_idxs) > self._k:
                raise RuntimeError('The algorithm selected more features than allowed.')
        return permissible_feature_idxs

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Run subgroup-discovery method

        Runs the subgroup-discovery method on the passed data, stores bounds internally, and
        returns meta-data about the fitting process.

        Parameters
        ----------
        X : pd.DataFrame
            Dataset (each row is a data object, each column a feature). All values must be numeric.
        y : pd.Series
            Prediction target. Must be boolean (false, true) or binary integer (0, 1) and have the
            same number of entries as `X` has rows.

        Returns
        -------
        Dict[str, Any]
            Meta-data about the fitting process. Contains the keys `objective_value` (WRacc),
            `optimization_time`, and `optimization_status`. The value of `optimization_status` is
            always `None` since the method always returns its local optimum.
        """

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
        beam_quality = np.array([wracc_np(y_true=y_np, y_pred=is_in_box)
                                 for is_in_box in beam_is_in_box])
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
