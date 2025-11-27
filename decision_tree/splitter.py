"""
Splitter module for finding optimal splits in Decision Tree.

This module contains the Splitter class that handles finding the best
split for each node during tree construction using VECTORIZED NumPy operations.

MODES (as per specification):
    - DETERMINISTIC (GBM mode): max_features=None or 1.0 -> all features
    - RANDOMIZED (RF mode): max_features='sqrt', int, float -> subset of features
"""

import numpy as np
from typing import Tuple, Optional
from .metrics import gini, entropy, mse, CRITERION_FUNCTIONS


class Splitter:
    """
    Splitter class for finding optimal splits using vectorized NumPy operations.
    
    This class encapsulates the logic for finding the best feature and threshold
    to split a node, supporting both deterministic (GBM) and randomized (RF) modes.
    
    Parameters
    ----------
    criterion : str
        Impurity criterion ('gini', 'entropy', or 'mse').
    min_samples_leaf : int
        Minimum samples required at a leaf node.
    random_state : np.random.RandomState or None
        Random state for reproducible feature sampling.
    
    Attributes
    ----------
    criterion : str
        The impurity criterion being used.
    min_samples_leaf : int
        Minimum samples at leaf nodes.
    impurity_func : callable
        The impurity function being used.
    """
    
    def __init__(
        self,
        criterion: str = 'gini',
        min_samples_leaf: int = 1,
        random_state: Optional[np.random.RandomState] = None
    ):
        """
        Initialize the Splitter.
        
        Parameters
        ----------
        criterion : str, default='gini'
            Impurity criterion ('gini', 'entropy', 'mse').
        min_samples_leaf : int, default=1
            Minimum samples at leaf.
        random_state : np.random.RandomState or None, default=None
            Random state for feature sampling.
        """
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        
        # Get impurity function
        if criterion not in CRITERION_FUNCTIONS:
            raise ValueError(f"Unknown criterion: {criterion}")
        self.impurity_func = CRITERION_FUNCTIONS[criterion]
    
    def _compute_split_gain_vectorized(
        self,
        y: np.ndarray,
        feature_values: np.ndarray,
        thresholds: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute information gain for all thresholds using vectorized operations.
        
        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            Target values.
        feature_values : np.ndarray of shape (n_samples,)
            Feature values for all samples.
        thresholds : np.ndarray of shape (n_thresholds,)
            Candidate threshold values.
        
        Returns
        -------
        gains : np.ndarray of shape (n_thresholds,)
            Information gain for each threshold.
        n_left : np.ndarray of shape (n_thresholds,)
            Number of samples in left child for each threshold.
        n_right : np.ndarray of shape (n_thresholds,)
            Number of samples in right child for each threshold.
        """
        n_samples = len(y)
        n_thresholds = len(thresholds)
        
        if n_thresholds == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Parent impurity (constant for all thresholds)
        parent_impurity = self.impurity_func(y)
        
        # Create mask matrix: (n_thresholds, n_samples)
        # left_masks[i, j] = True if sample j goes to left child for threshold i
        left_masks = feature_values.reshape(1, -1) <= thresholds.reshape(-1, 1)
        
        # Count samples in each split
        n_left = np.sum(left_masks, axis=1)
        n_right = n_samples - n_left
        
        # Initialize gains array
        gains = np.full(n_thresholds, -np.inf)
        
        # Compute gain for each valid threshold
        for i in range(n_thresholds):
            # Check min_samples_leaf constraint
            if n_left[i] < self.min_samples_leaf or n_right[i] < self.min_samples_leaf:
                continue
            
            # Get left and right targets
            mask = left_masks[i]
            y_left = y[mask]
            y_right = y[~mask]
            
            # Compute weighted impurity
            left_impurity = self.impurity_func(y_left)
            right_impurity = self.impurity_func(y_right)
            
            weighted_impurity = (n_left[i] / n_samples) * left_impurity + \
                               (n_right[i] / n_samples) * right_impurity
            
            # Information gain
            gains[i] = parent_impurity - weighted_impurity
        
        return gains, n_left, n_right
    
    def find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_indices: np.ndarray
    ) -> Tuple[int, float, float]:
        """
        Find the best split for a node.
        
        This method searches through all candidate features and thresholds
        to find the split that maximizes information gain.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix for the current node.
        y : np.ndarray of shape (n_samples,)
            Target values for the current node.
        feature_indices : np.ndarray
            Indices of features to consider (depends on mode).
        
        Returns
        -------
        best_feature : int
            Index of the best feature to split on. -1 if no valid split.
        best_threshold : float
            Best threshold value for the split.
        best_gain : float
            Information gain from the best split. -inf if no valid split.
        """
        n_samples = len(y)
        
        best_feature = -1
        best_threshold = 0.0
        best_gain = -np.inf
        
        # Search through all candidate features
        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]
            
            # Get unique values for this feature
            unique_values = np.unique(feature_values)
            
            if len(unique_values) <= 1:
                # Cannot split on constant feature
                continue
            
            # Compute candidate thresholds as midpoints between consecutive unique values
            # This is more efficient than using all unique values
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0
            
            # Compute gains for all thresholds (vectorized)
            gains, _, _ = self._compute_split_gain_vectorized(y, feature_values, thresholds)
            
            if len(gains) == 0:
                continue
            
            # Find best threshold for this feature
            max_idx = np.argmax(gains)
            max_gain = gains[max_idx]
            
            if max_gain > best_gain:
                best_gain = max_gain
                best_feature = feature_idx
                best_threshold = float(thresholds[max_idx])
        
        return best_feature, best_threshold, best_gain
    
    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_index: int,
        threshold: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data based on feature and threshold.
        
        Samples with feature value <= threshold go to left child,
        samples with feature value > threshold go to right child.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : np.ndarray of shape (n_samples,)
            Target values.
        feature_index : int
            Index of the feature to split on.
        threshold : float
            Threshold value for the split.
        
        Returns
        -------
        X_left : np.ndarray
            Features for left child.
        y_left : np.ndarray
            Targets for left child.
        X_right : np.ndarray
            Features for right child.
        y_right : np.ndarray
            Targets for right child.
        """
        left_mask = X[:, feature_index] <= threshold
        
        return X[left_mask], y[left_mask], X[~left_mask], y[~left_mask]
