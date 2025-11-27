"""
CART-style Decision Tree Implementation.

This module contains the main DecisionTree class for classification and regression.
It is designed to be used as a base learner for Random Forest and GBM implementations
following Breiman (2001) specifications.

TWO MODES:
    - DETERMINISTIC (GBM mode): max_features=None -> Search ALL features at every split
    - RANDOMIZED (RF mode): max_features='sqrt'/int/float -> Sample features per split

CLASS SIGNATURE (DO NOT CHANGE - Required by teammates):
    DecisionTree(max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 criterion='gini', max_features=None, random_state=None)
"""

import numpy as np
from typing import Optional, Union

from .node import Node
from .splitter import Splitter
from .metrics import gini, entropy, mse, CRITERION_FUNCTIONS


class DecisionTree:
    """
    CART-style Decision Tree for classification and regression.
    
    This implementation supports both classification (Gini impurity, Entropy)
    and regression (MSE) tasks. It can be used directly or as a base estimator
    for ensemble methods like Random Forest (Breiman 2001) and Gradient Boosting.
    
    Parameters
    ----------
    max_depth : int or None, default=None
        Maximum depth of the tree. None means unlimited (unpruned trees as per Breiman).
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.
    criterion : str, default='gini'
        The function to measure the quality of a split.
        - 'gini': Gini impurity (classification)
        - 'entropy': Information entropy (classification)
        - 'mse': Mean squared error (regression)
    max_features : int, float, str or None, default=None
        The number of features to consider when looking for the best split:
        - None or 1.0: All features (DETERMINISTIC mode for GBM)
        - 'sqrt': sqrt(n_features) (RANDOMIZED mode for RF, Breiman default)
        - int: Exact number of features
        - float in (0,1): Fraction of features
    random_state : int or None, default=None
        Seed for reproducible randomized feature selection.
    
    Attributes
    ----------
    root_ : Node
        The root node of the fitted tree. Can be accessed by RF/GBM.
    n_features_ : int
        Number of features in the training data.
    n_classes_ : int
        Number of classes (classification only).
    classes_ : ndarray
        Unique class labels (classification only).
    task_type_ : str
        'classification' or 'regression'.
    feature_importances_ : ndarray
        Feature importances based on impurity decrease.
    
    Examples
    --------
    Classification (GBM mode - deterministic, all features):
    >>> tree = DecisionTree(max_depth=3, criterion='gini', max_features=None)
    >>> tree.fit(X_train, y_train)
    >>> predictions = tree.predict(X_test)
    
    Classification (RF mode - randomized, feature subset):
    >>> tree = DecisionTree(max_depth=None, criterion='gini', 
    ...                     max_features='sqrt', random_state=42)
    >>> tree.fit(X_train, y_train)
    >>> predictions = tree.predict(X_test)
    
    Regression:
    >>> tree = DecisionTree(criterion='mse', max_depth=5)
    >>> tree.fit(X_train, y_train)
    >>> predictions = tree.predict(X_test)
    """
    
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = 'gini',
        max_features: Optional[Union[int, float, str]] = None,
        random_state: Optional[int] = None
    ):
        """
        Initialize the DecisionTree.
        
        Parameters
        ----------
        max_depth : int or None, default=None
            Maximum depth of the tree. None for unlimited.
        min_samples_split : int, default=2
            Minimum samples required to split a node.
        min_samples_leaf : int, default=1
            Minimum samples required at a leaf node.
        criterion : str, default='gini'
            Splitting criterion ('gini', 'entropy', or 'mse').
        max_features : int, float, str or None, default=None
            Number of features to consider at each split.
        random_state : int or None, default=None
            Random seed for reproducibility.
        """
        # Store parameters EXACTLY as specified
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state
        
        # Attributes set during fit
        self.root_ = None
        self.n_features_ = None
        self.n_classes_ = None
        self.classes_ = None
        self.task_type_ = None
        self.feature_importances_ = None
        self._rng = None
        self._splitter = None
    
    def _determine_task_type(self, y: np.ndarray) -> str:
        """
        Auto-detect task type based on target values.
        
        Classification if:
        - y has <= 20 unique values AND
        - y is not floating point dtype
        
        Parameters
        ----------
        y : ndarray
            Target values.
        
        Returns
        -------
        str
            'classification' or 'regression'
        """
        # Use criterion as primary hint
        if self.criterion in ['gini', 'entropy']:
            return 'classification'
        elif self.criterion == 'mse':
            return 'regression'
        
        # Auto-detect based on target values
        unique_values = np.unique(y)
        n_unique = len(unique_values)
        is_float = np.issubdtype(y.dtype, np.floating)
        
        if n_unique <= 20 and not is_float:
            return 'classification'
        else:
            return 'regression'
    
    def _get_n_features_to_sample(self) -> int:
        """
        Determine number of features to consider at each split.
        
        Returns
        -------
        int
            Number of features to sample.
        """
        if self.max_features is None:
            # Deterministic mode (GBM): use all features
            return self.n_features_
        elif self.max_features == 'sqrt':
            # RF default for classification (Breiman 2001)
            return max(1, int(np.sqrt(self.n_features_)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(self.n_features_)))
        elif isinstance(self.max_features, int):
            return min(max(1, self.max_features), self.n_features_)
        elif isinstance(self.max_features, float):
            if self.max_features >= 1.0:
                # 1.0 means all features (deterministic)
                return self.n_features_
            return max(1, int(self.max_features * self.n_features_))
        else:
            return self.n_features_
    
    def _get_candidate_features(self) -> np.ndarray:
        """
        Get feature indices to consider for current split.
        
        Implements two modes:
        - Deterministic: return all features (0 to n_features)
        - Randomized: sample features WITHOUT replacement
        
        Returns
        -------
        ndarray
            Array of feature indices to consider.
        """
        n_to_sample = self._get_n_features_to_sample()
        
        if n_to_sample >= self.n_features_:
            # Deterministic mode: consider ALL features
            return np.arange(self.n_features_)
        else:
            # Randomized mode: sample features WITHOUT replacement
            return self._rng.choice(
                self.n_features_,
                size=n_to_sample,
                replace=False
            )
    
    def _compute_leaf_value(self, y: np.ndarray) -> Union[int, float]:
        """
        Compute prediction value for a leaf node.
        
        Classification: argmax(class_counts) - majority class
        Regression: np.mean(y) - mean of targets
        
        Parameters
        ----------
        y : ndarray
            Target values at the leaf.
        
        Returns
        -------
        int or float
            Leaf prediction value.
        """
        if self.task_type_ == 'classification':
            # Return majority class (argmax of class counts)
            values, counts = np.unique(y, return_counts=True)
            return values[np.argmax(counts)]
        else:
            # Return mean for regression
            return float(np.mean(y))
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """
        Recursively build the decision tree.
        
        Stopping conditions (ALL implemented as per spec):
        1. max_depth reached
        2. n_samples < min_samples_split
        3. Pure node (all same class/value)
        4. n_samples < 2 * min_samples_leaf (can't create valid children)
        5. best_gain <= 0 (no improvement possible)
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : ndarray of shape (n_samples,)
            Target values.
        depth : int
            Current depth from root.
        
        Returns
        -------
        Node
            The root node of the (sub)tree.
        """
        n_samples = len(y)
        
        # Compute current impurity for feature importance
        current_impurity = self._splitter.impurity_func(y) if n_samples > 0 else 0.0
        
        # === STOPPING CONDITIONS ===
        
        # 1. Max depth reached
        if self.max_depth is not None and depth >= self.max_depth:
            return Node(feature_index=-1, leaf_value=self._compute_leaf_value(y))
        
        # 2. Not enough samples to split
        if n_samples < self.min_samples_split:
            return Node(feature_index=-1, leaf_value=self._compute_leaf_value(y))
        
        # 3. Pure node
        unique_values = np.unique(y)
        if len(unique_values) == 1:
            return Node(feature_index=-1, leaf_value=self._compute_leaf_value(y))
        
        # 4. Cannot create valid children with min_samples_leaf
        if n_samples < 2 * self.min_samples_leaf:
            return Node(feature_index=-1, leaf_value=self._compute_leaf_value(y))
        
        # === FIND BEST SPLIT ===
        
        # Get candidate features (ALL for deterministic, SUBSET for randomized)
        candidate_features = self._get_candidate_features()
        
        # Find the best split
        best_feature, best_threshold, best_gain = self._splitter.find_best_split(
            X, y, candidate_features
        )
        
        # 5. No positive gain possible
        if best_gain <= 0 or best_feature == -1:
            return Node(feature_index=-1, leaf_value=self._compute_leaf_value(y))
        
        # === PERFORM SPLIT ===
        
        # Split the data
        X_left, y_left, X_right, y_right = self._splitter.split_data(
            X, y, best_feature, best_threshold
        )
        
        # Update feature importance (weighted impurity decrease)
        n_left, n_right = len(y_left), len(y_right)
        left_impurity = self._splitter.impurity_func(y_left)
        right_impurity = self._splitter.impurity_func(y_right)
        
        importance = (n_samples * current_impurity - 
                     n_left * left_impurity - 
                     n_right * right_impurity)
        self.feature_importances_[best_feature] += importance
        
        # === RECURSE ===
        
        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)
        
        return Node(
            feature_index=best_feature,
            threshold=best_threshold,
            left_child=left_child,
            right_child=right_child
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTree':
        """
        Build the decision tree from training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Target values (class labels or continuous values).
        
        Returns
        -------
        self : DecisionTree
            Fitted estimator (for method chaining).
        """
        # Convert to numpy arrays
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        
        # Handle 1D input
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Validate shapes
        if len(X) != len(y):
            raise ValueError(f"X and y must have same number of samples. "
                           f"Got X: {len(X)}, y: {len(y)}")
        
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        
        # Initialize random state for reproducibility
        if self.random_state is not None:
            self._rng = np.random.RandomState(self.random_state)
        else:
            self._rng = np.random.RandomState()
        
        # Auto-detect task type
        self.task_type_ = self._determine_task_type(y)
        
        # Store class information for classification
        if self.task_type_ == 'classification':
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
        else:
            self.classes_ = None
            self.n_classes_ = None
        
        # Determine effective criterion
        effective_criterion = self.criterion
        if self.task_type_ == 'regression' and self.criterion in ['gini', 'entropy']:
            effective_criterion = 'mse'
        
        # Initialize splitter
        self._splitter = Splitter(
            criterion=effective_criterion,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self._rng
        )
        
        # Initialize feature importances
        self.feature_importances_ = np.zeros(n_features)
        
        # Build tree (unpruned as per Breiman 2001)
        self.root_ = self._build_tree(X, y, depth=0)
        
        # Normalize feature importances
        total = np.sum(self.feature_importances_)
        if total > 0:
            self.feature_importances_ /= total
        
        return self
    
    def _predict_sample(self, x: np.ndarray, node: Node) -> Union[int, float]:
        """
        Traverse tree to predict for a single sample.
        
        Parameters
        ----------
        x : ndarray of shape (n_features,)
            Single sample.
        node : Node
            Current node.
        
        Returns
        -------
        prediction
            Leaf value.
        """
        # Leaf node
        if node.is_leaf():
            return node.leaf_value
        
        # Decision node: go left if feature <= threshold, else right
        if x[node.feature_index] <= node.threshold:
            return self._predict_sample(x, node.left_child)
        else:
            return self._predict_sample(x, node.right_child)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels or values for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels (classification) or values (regression).
        """
        if self.root_ is None:
            raise ValueError("Tree has not been fitted. Call fit() first.")
        
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        predictions = np.array([self._predict_sample(x, self.root_) for x in X])
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X.
        
        Note: For a single tree, this returns hard predictions (0 or 1).
        More useful in ensemble methods where probabilities are averaged.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
        
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        if self.task_type_ != 'classification':
            raise ValueError("predict_proba only available for classification.")
        
        if self.root_ is None:
            raise ValueError("Tree has not been fitted. Call fit() first.")
        
        predictions = self.predict(X)
        n_samples = len(predictions)
        proba = np.zeros((n_samples, self.n_classes_))
        
        for i, pred in enumerate(predictions):
            class_idx = np.where(self.classes_ == pred)[0][0]
            proba[i, class_idx] = 1.0
        
        return proba
    
    def get_depth(self) -> int:
        """
        Get the maximum depth of the tree.
        
        Returns
        -------
        int
            Tree depth (0 for single leaf).
        """
        if self.root_ is None:
            return 0
        
        def _depth(node: Node) -> int:
            if node is None or node.is_leaf():
                return 0
            return 1 + max(_depth(node.left_child), _depth(node.right_child))
        
        return _depth(self.root_)
    
    def get_n_leaves(self) -> int:
        """
        Get the number of leaf nodes.
        
        Returns
        -------
        int
            Number of leaves.
        """
        if self.root_ is None:
            return 0
        
        def _count(node: Node) -> int:
            if node is None:
                return 0
            if node.is_leaf():
                return 1
            return _count(node.left_child) + _count(node.right_child)
        
        return _count(self.root_)
    
    def __repr__(self) -> str:
        """String representation."""
        params = []
        if self.max_depth is not None:
            params.append(f"max_depth={self.max_depth}")
        if self.min_samples_split != 2:
            params.append(f"min_samples_split={self.min_samples_split}")
        if self.min_samples_leaf != 1:
            params.append(f"min_samples_leaf={self.min_samples_leaf}")
        if self.criterion != 'gini':
            params.append(f"criterion='{self.criterion}'")
        if self.max_features is not None:
            params.append(f"max_features={self.max_features!r}")
        if self.random_state is not None:
            params.append(f"random_state={self.random_state}")
        
        return f"DecisionTree({', '.join(params)})"
