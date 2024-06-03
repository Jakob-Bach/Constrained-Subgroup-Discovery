"""Subgroup-discovery metrics

Functions for subgroup-discovery evaluation metrics, quantifying subgroup quality and similarity.
"""


from typing import Any, Sequence, Union

import numpy as np


def wracc(y_true: Sequence[Union[bool, int]], y_pred: Sequence[Union[bool, int]]) -> float:
    """Weighted relative accuracy

    Computes the weighted relative accuracy (WRAcc) for two binary (bool or int) sequences (may
    also be :class:`pd.Series` or :class:`np.array`) indicating class labels and predictions.
    The range of WRAcc is at most [-0.25, 0.25] but actually depends on the class imbalance
    (becomes smaller if the classes are more imbalanced).

    Literature
    ----------
    Lavrac et al. (1999): "Rule Evaluation Measures: A Unifying View"

    Parameters
    ----------
    y_true : Sequence[Union[bool, int]]
        Binary ground-truth labels.
    y_pred : Sequence[Union[bool, int]]
        Binary predicted labels.

    Returns
    -------
    float
        Value of the WRAcc metric.
    """
    n_true_pos = sum(y_t and y_p for y_t, y_p in zip(y_true, y_pred))
    n_instances = len(y_true)
    n_actual_pos = sum(y_true)
    n_pred_pos = sum(y_pred)
    return n_true_pos / n_instances - n_pred_pos * n_actual_pos / (n_instances ** 2)


def wracc_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted relative accuracy

    Same functionality as :func:`wracc`, but faster and intended for binary (bool or int) `numpy`
    arrays. This fast method should be preferred if called often as a subroutine.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground-truth labels.
    y_pred : np.ndarray
        Binary predicted labels.

    Returns
    -------
    float
        Value of the WRAcc metric.
    """
    n_true_pos = np.count_nonzero(y_true & y_pred)
    n_instances = len(y_true)
    n_actual_pos = np.count_nonzero(y_true)
    n_pred_pos = np.count_nonzero(y_pred)
    return n_true_pos / n_instances - n_pred_pos * n_actual_pos / (n_instances ** 2)


def nwracc(y_true: Sequence[Union[bool, int]], y_pred: Sequence[Union[bool, int]]) -> float:
    """Normalized weighted relative accuracy

    Computes the normalized weighted relative accuracy (nWRAcc) for two binary (bool or int)
    sequences (may also be :class:`pd.Series` or :class:`np.array`) indicating class labels and
    predictions. This metric equals WRAcc (:func:`wracc`) divided by its maximum (= product of
    class probabilities) and therefore always has the range [-1, 1], no matter how imbalanced the
    two classes are.

    Literature
    ----------
    Mathonat et al. (2021): "Anytime Subgroup Discovery in High Dimensional Numerical Data"

    Parameters
    ----------
    y_true : Sequence[Union[bool, int]]
        Binary ground-truth labels.
    y_pred : Sequence[Union[bool, int]]
        Binary prediced labels.

    Returns
    -------
    float
        Value of the nWRAcc metric..
    """
    n_true_pos = sum(y_t and y_p for y_t, y_p in zip(y_true, y_pred))
    n_instances = len(y_true)
    n_actual_pos = sum(y_true)
    n_pred_pos = sum(y_pred)
    enumerator = n_true_pos * n_instances - n_pred_pos * n_actual_pos
    denominator = n_actual_pos * (n_instances - n_actual_pos)
    return enumerator / denominator


def nwracc_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized weighted relative accuracy

    Same functionality as :func:`nwracc`, but faster and intended for binary (bool or int) `numpy`
    arrays. This fast method should be preferred if called often as a subroutine.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground-truth labels.
    y_pred : np.ndarray
        Binary predicted labels.

    Returns
    -------
    float
        Value of the nWRAcc metric.
    """
    n_true_pos = np.count_nonzero(y_true & y_pred)
    n_instances = len(y_true)
    n_actual_pos = np.count_nonzero(y_true)
    n_pred_pos = np.count_nonzero(y_pred)
    enumerator = n_true_pos * n_instances - n_pred_pos * n_actual_pos
    denominator = n_actual_pos * (n_instances - n_actual_pos)
    return enumerator / denominator


def jaccard(set_1_indicators: Sequence[Union[bool, int]],
            set_2_indicators: Sequence[Union[bool, int]]) -> float:
    """Jaccard similarity

    Computes the Jaccard similarity between two binary (bool or int) sequences (may also be
    :class:`pd.Series` or :class:`np.array`) indicating set membership. It is a symmetric measure
    (i.e., order of arguments does not matter) with the range [0, 1].

    Literature
    ----------
    https://en.wikipedia.org/wiki/Jaccard_index

    Parameters
    ----------
    set_1_indicators : Sequence[Union[bool, int]]
        Binary membership indicators for elements in first set.
    set_2_indicators : Sequence[Union[bool, int]]
        Binary membership indicators for elements in second set.

    Returns
    -------
    float
        Value of the Jaccard similarity. NaN if both sets are empty.
    """
    size_intersection = sum(i_1 and i_2 for i_1, i_2 in zip(set_1_indicators, set_2_indicators))
    size_union = sum(i_1 or i_2 for i_1, i_2 in zip(set_1_indicators, set_2_indicators))
    return size_intersection / size_union if size_union != 0 else float('nan')


def jaccard_np(set_1_indicators: np.ndarray, set_2_indicators: np.ndarray) -> float:
    """Jaccard similarity

    Same functionality as :func:`jaccard`, but faster and intended for binary (bool or int) `numpy`
    arrays. This fast method should be preferred if called often as a subroutine.

    Parameters
    ----------
    set_1_indicators : np.ndarray
        Binary membership indicators for elements in first set.
    set_2_indicators : np.ndarray
        Binary membership indicators for elements in second set.

    Returns
    -------
    float
        Value of the Jaccard similarity. NaN if both sets are empty.
    """
    size_intersection = np.count_nonzero(set_1_indicators & set_2_indicators)
    size_union = np.count_nonzero(set_1_indicators | set_2_indicators)
    return size_intersection / size_union if size_union != 0 else float('nan')


def hamming(sequence_1: Sequence[Any], sequence_2: Sequence[Any]) -> float:
    """Normalized hamming similarity

    Computes the normalized Hamming similarity, i.e., 1 - Hamming distance normalized to [0, 1],
    between two sequences (may also be :class:`pd.Series` or class:`np.array`). It is a symmetric
    measure (i.e., order of arguments does not matter) with the range [0, 1]. Since it only checks
    whether elements are identical or not, the elements may be of arbitrary type. For two binary
    vectors, it equals prediction accuracy.

    Literature
    ----------
    https://en.wikipedia.org/wiki/Hamming_distance

    Parameters
    ----------
    sequence_1 : Sequence[Any]
        First sequence of elements.
    sequence_2 : Sequence[Any]
        Second sequence of elements.

    Returns
    -------
    float
        Value of the normalized Hamming similarity.
    """
    size_identical = sum(s_1 == s_2 for s_1, s_2 in zip(sequence_1, sequence_2))
    return size_identical / len(sequence_1)


def hamming_np(sequence_1: np.array, sequence_2: np.array) -> float:
    """Normalized hamming similarity

    Same functionality as :func:`hamming`, but faster and intended for `numpy` arrays. This fast
    method should be preferred if called often as a subroutine.

    Parameters
    ----------
    sequence_1 : np.array
        First sequence of elements.
    sequence_2 : np.array
        Second sequence of elements.

    Returns
    -------
    float
        Value of the normalized Hamming similarity.
    """
    size_identical = (sequence_1 == sequence_2).sum()
    return size_identical / len(sequence_1)
