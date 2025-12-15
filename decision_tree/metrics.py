"""
Impurity and error metrics for Decision Tree splitting.

This module contains functions for computing impurity measures used
in classification and regression tree building.

EXACT FORMULAS (as per specification):
    - gini(p): sum(p_i * (1 - p_i)) = 1 - sum(p_i^2)
    - entropy(p): -sum(p_i * log2(p_i))
    - mse(y): mean((y - mean(y))^2)

These functions are used internally by the DecisionTree class and
can also be imported for testing purposes.
"""

import numpy as np
from typing import Union, List


def gini(y: np.ndarray) -> float:
    """
    Compute Gini impurity for a set of labels.
    
    Gini impurity measures the probability of incorrectly classifying
    a randomly chosen element. Formula: sum(p_i * (1 - p_i)) = 1 - sum(p_i^2)
    
    Parameters
    ----------
    y : ndarray
        Array of class labels.
    
    Returns
    -------
    float
        Gini impurity value in range [0, 1 - 1/n_classes].
        0 indicates perfect purity (all samples belong to one class).
    
    Examples
    --------
    >>> gini(np.array([0, 0, 1, 1]))
    0.5
    >>> gini(np.array([0, 0, 0]))
    0.0
    >>> gini(np.array([]))
    0.0
    """
    y = np.asarray(y)
    if len(y) == 0:
        return 0.0
    
    _, counts = np.unique(y, return_counts=True)
    proportions = counts / len(y)
    return float(1.0 - np.sum(proportions ** 2))


def gini_from_proportions(p: Union[List[float], np.ndarray]) -> float:
    """
    Compute Gini impurity directly from class proportions.
    
    Formula: sum(p_i * (1 - p_i))
    
    Parameters
    ----------
    p : array-like
        Array of class proportions (should sum to 1).
    
    Returns
    -------
    float
        Gini impurity value.
    
    Examples
    --------
    >>> gini_from_proportions([0.5, 0.5])
    0.5
    >>> gini_from_proportions([1.0, 0.0])
    0.0
    >>> gini_from_proportions([0.25, 0.25, 0.25, 0.25])
    0.75
    """
    p = np.asarray(p, dtype=np.float64)
    return float(np.sum(p * (1 - p)))


def entropy(y: np.ndarray) -> float:
    """
    Compute information entropy for a set of labels.
    
    Entropy measures the average amount of information (uncertainty)
    in the class distribution. Formula: -sum(p_i * log2(p_i))
    
    Parameters
    ----------
    y : ndarray
        Array of class labels.
    
    Returns
    -------
    float
        Entropy value in range [0, log2(n_classes)].
        0 indicates perfect purity (all samples belong to one class).
    
    Examples
    --------
    >>> entropy(np.array([0, 0, 1, 1]))
    1.0
    >>> entropy(np.array([0, 0, 0]))
    0.0
    >>> abs(entropy(np.array([0, 1, 2])) - 1.5849625007211563) < 1e-6
    True
    """
    y = np.asarray(y)
    if len(y) == 0:
        return 0.0
    
    _, counts = np.unique(y, return_counts=True)
    proportions = counts / len(y)
    # Filter out zero proportions to avoid log(0)
    proportions = proportions[proportions > 0]
    
    return float(-np.sum(proportions * np.log2(proportions)))


def entropy_from_proportions(p: Union[List[float], np.ndarray]) -> float:
    """
    Compute entropy directly from class proportions.
    
    Formula: -sum(p_i * log2(p_i))
    
    Parameters
    ----------
    p : array-like
        Array of class proportions (should sum to 1).
    
    Returns
    -------
    float
        Entropy value.
    
    Examples
    --------
    >>> entropy_from_proportions([0.5, 0.5])
    1.0
    >>> abs(entropy_from_proportions([1/3, 1/3, 1/3]) - np.log2(3)) < 1e-6
    True
    """
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]  # Filter zeros to avoid log(0)
    return float(-np.sum(p * np.log2(p)))


def mse(y: np.ndarray) -> float:
    """
    Compute Mean Squared Error (variance) for regression.
    
    MSE measures the average squared deviation from the mean.
    Formula: mean((y - mean(y))^2)
    
    Parameters
    ----------
    y : ndarray
        Array of target values.
    
    Returns
    -------
    float
        MSE (variance) of the target values.
        0 indicates all values are identical.
    
    Examples
    --------
    >>> abs(mse(np.array([1, 3, 5])) - 8/3) < 1e-6
    True
    >>> mse(np.array([5, 5, 5]))
    0.0
    >>> mse(np.array([]))
    0.0
    """
    y = np.asarray(y)
    if len(y) == 0:
        return 0.0
    return float(np.mean((y - np.mean(y)) ** 2))


def information_gain(
    y_parent: np.ndarray,
    y_left: np.ndarray,
    y_right: np.ndarray,
    criterion: str = 'gini'
) -> float:
    """
    Compute information gain from a split.
    
    Information gain is the reduction in impurity achieved by
    splitting the parent node into left and right children.
    
    Formula: gain = I_parent - (n_left/n * I_left + n_right/n * I_right)
    
    Parameters
    ----------
    y_parent : ndarray
        Target values for the parent node.
    y_left : ndarray
        Target values for the left child.
    y_right : ndarray
        Target values for the right child.
    criterion : str, default='gini'
        Impurity criterion ('gini', 'entropy', or 'mse').
    
    Returns
    -------
    float
        Information gain value. Higher is better.
    
    Examples
    --------
    >>> y_parent = np.array([0, 0, 1, 1])
    >>> y_left = np.array([0, 0])
    >>> y_right = np.array([1, 1])
    >>> information_gain(y_parent, y_left, y_right, 'gini')
    0.5
    """
    n = len(y_parent)
    n_left = len(y_left)
    n_right = len(y_right)
    
    if n == 0 or n_left == 0 or n_right == 0:
        return 0.0
    
    # Select impurity function
    impurity_funcs = {
        'gini': gini,
        'entropy': entropy,
        'mse': mse
    }
    
    if criterion not in impurity_funcs:
        raise ValueError(f"Unknown criterion: {criterion}. Use 'gini', 'entropy', or 'mse'.")
    
    impurity_func = impurity_funcs[criterion]
    
    # Compute information gain
    parent_impurity = impurity_func(y_parent)
    left_impurity = impurity_func(y_left)
    right_impurity = impurity_func(y_right)
    
    weighted_child_impurity = (n_left / n) * left_impurity + (n_right / n) * right_impurity
    
    return float(parent_impurity - weighted_child_impurity)


# Mapping from criterion names to functions (for external use)
CRITERION_FUNCTIONS = {
    'gini': gini,
    'entropy': entropy,
    'mse': mse
}
