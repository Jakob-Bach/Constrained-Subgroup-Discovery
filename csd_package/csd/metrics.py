"""Subgroup-discovery metrics

Functions for subgroup-discovery evaluation metrics.
"""


from typing import Any, Sequence

import numpy as np


# Compute the weighted relative accuracy (WRAcc) for two binary (bool or int) sequences (may also
# be pd.Series or np.array) indicating class labels and predictions. The range of WRAcc is at most
# [-0.25, 0.25] but actually depends on the class imbalance (becomes smaller if the classes are
# more imbalanced).
# Literature: Lavrac et al. (1999): "Rule Evaluation Measures: A Unifying View"
def wracc(y_true: Sequence[bool], y_pred: Sequence[bool]) -> float:
    n_true_pos = sum(y_t and y_p for y_t, y_p in zip(y_true, y_pred))
    n_instances = len(y_true)
    n_actual_pos = sum(y_true)
    n_pred_pos = sum(y_pred)
    return n_true_pos / n_instances - n_pred_pos * n_actual_pos / (n_instances ** 2)


# Same functionality as wracc(), but faster and intended for binary (bool or int) numpy arrays.
# This fast method should be preferred if called often as a subroutine.
def wracc_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n_true_pos = np.count_nonzero(y_true & y_pred)
    n_instances = len(y_true)
    n_actual_pos = np.count_nonzero(y_true)
    n_pred_pos = np.count_nonzero(y_pred)
    return n_true_pos / n_instances - n_pred_pos * n_actual_pos / (n_instances ** 2)


# Compute the normalized weighted relative accuracy (nWRAcc) for two binary (bool or int)
# sequences (may also be pd.Series or np.array) indicating class labels and predictions.
# This metric equals WRAcc divided by its maximum (= product of class probabilities) and therefore
# always has the range [-1, 1], no matter how imbalanced the two classes are.
# Literature: Mathonat et al. (2021): "Anytime Subgroup Discovery in High Dimensional Numerical Data"
def nwracc(y_true: Sequence[bool], y_pred: Sequence[bool]) -> float:
    n_true_pos = sum(y_t and y_p for y_t, y_p in zip(y_true, y_pred))
    n_instances = len(y_true)
    n_actual_pos = sum(y_true)
    n_pred_pos = sum(y_pred)
    enumerator = n_true_pos * n_instances - n_pred_pos * n_actual_pos
    denominator = n_actual_pos * (n_instances - n_actual_pos)
    return enumerator / denominator


# Same functionality as nwracc(), but faster and intended for binary (bool or int) numpy arrays.
# This fast method should be preferred if called often as a subroutine.
def nwracc_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n_true_pos = np.count_nonzero(y_true & y_pred)
    n_instances = len(y_true)
    n_actual_pos = np.count_nonzero(y_true)
    n_pred_pos = np.count_nonzero(y_pred)
    enumerator = n_true_pos * n_instances - n_pred_pos * n_actual_pos
    denominator = n_actual_pos * (n_instances - n_actual_pos)
    return enumerator / denominator


# Compute the Jaccard similarity between two binary (bool or int) sequences (may also be pd.Series
# or np.array) indicating set membership. Its range is [0, 1].
def jaccard(set_1_indicators: Sequence[bool], set_2_indicators: Sequence[bool]) -> float:
    size_intersection = sum(i_1 and i_2 for i_1, i_2 in zip(set_1_indicators, set_2_indicators))
    size_union = sum(i_1 or i_2 for i_1, i_2 in zip(set_1_indicators, set_2_indicators))
    return size_intersection / size_union if size_union != 0 else float('nan')


# Same functionality as jaccard(), but faster and intended for binary (bool or int) numpy arrays.
# This fast method should be preferred if called often as a subroutine.
def jaccard_np(set_1_indicators: np.ndarray, set_2_indicators: np.ndarray) -> float:
    size_intersection = np.count_nonzero(set_1_indicators & set_2_indicators)
    size_union = np.count_nonzero(set_1_indicators | set_2_indicators)
    return size_intersection / size_union if size_union != 0 else float('nan')


# Compute the Hamming similarity, normalized to [0, 1], between two sequences (may also be
# pd.Series or np.array). Its range is [0, 1].
def hamming(sequence_1: Sequence[Any], sequence_2: Sequence[Any]) -> float:
    size_identical = sum(s_1 == s_2 for s_1, s_2 in zip(sequence_1, sequence_2))
    return size_identical / len(sequence_1)


# Same functionality as hamming(), but faster and intended for numpy arrays.
def hamming_np(sequence_1: np.array, sequence_2: np.array) -> float:
    size_identical = (sequence_1 == sequence_2).sum()
    return size_identical / len(sequence_1)
