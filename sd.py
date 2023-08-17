"""Subgroup discovery

Classes for subgroup-discovery methods.
"""


from abc import ABCMeta, abstractmethod
from typing import Dict

import pandas as pd


# Computes the weighted relative accuracy (WRAcc) for two binary (bool or int) series.
def wracc(y_true: pd.Series, y_pred: pd.Series) -> float:
    assert len(y_true) == len(y_pred), "Prediction and ground truth need to have same length."
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
