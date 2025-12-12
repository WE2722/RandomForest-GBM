"""
Random Forest Implementation (Member 2)

Based on Breiman (2001): "Random Forests"

This is a simple implementation of Random Forest that uses the DecisionTree class
as base learners. It follows the key principles:
1. Bootstrap sampling of training data
2. Random feature selection at each split
3. Averaging predictions (regression) or voting (classification)
"""

import numpy as np
import sys
sys.path.insert(0, '..')
from decision_tree import DecisionTree


class RandomForest:
    """
    Random Forest Classifier/Regressor based on Breiman (2001).
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    
    max_depth : int or None, default=None
        Maximum depth of each tree. None means unlimited.
    
    min_samples_split : int, default=2
        Minimum samples required to split a node.
    
    min_samples_leaf : int, default=1
        Minimum samples required in a leaf node.
    
    max_features : str, int, or float, default='sqrt'
        Number of features to consider at each split:
        - 'sqrt': sqrt(n_features) - RECOMMENDED for classification
        - 'log2': log2(n_features)
        - int: exact number
        - float: fraction of n_features
    
    bootstrap : bool, default=True
        Whether to use bootstrap sampling. MUST be True for Random Forest.
    
    oob_score : bool, default=False
        Whether to compute out-of-bag score.
    
    random_state : int or None, default=None
        Random seed for reproducibility.
    
    n_jobs : int or None, default=None
        Number of parallel jobs (not implemented yet).
    
    Attributes
    ----------
    estimators_ : list of DecisionTree
        The collection of fitted trees.
    
    feature_importances_ : ndarray of shape (n_features,)
        Feature importances (averaged across trees).
    
    oob_score_ : float
        Out-of-bag score (if oob_score=True).
    
    n_features_ : int
        Number of features.
    
    n_classes_ : int
        Number of classes (classification only).
    
    classes_ : ndarray
        Class labels (classification only).
    """
    
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        oob_score=False,
        random_state=None,
        n_jobs=None
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Attributes initialized after fit()
        self.estimators_ = []
        self.feature_importances_ = None
        self.oob_score_ = None
        self.n_features_ = None
        self.n_classes_ = None
        self.classes_ = None
        self.task_type_ = None
        self._rng = None
    
    def _determine_task_type(self, y):
        """Determine if this is classification or regression."""
        unique_values = np.unique(y)
        n_unique = len(unique_values)
        is_float = np.issubdtype(y.dtype, np.floating)
        
        if n_unique <= 20 and not is_float:
            return 'classification'
        else:
            return 'regression'
    
    def fit(self, X, y):
        """Build a forest of trees from training data using bootstrap sampling."""
        # Convert to numpy arrays
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        
        # Handle 1D input
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        
        # Initialize random state
        if self.random_state is not None:
            self._rng = np.random.RandomState(self.random_state)
        else:
            self._rng = np.random.RandomState()
        
        # Determine task type (classification or regression)
        self.task_type_ = self._determine_task_type(y)
        
        # Store class information for classification
        if self.task_type_ == 'classification':
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
        else:
            self.classes_ = None
            self.n_classes_ = None
        
        # Initialize estimators list
        self.estimators_ = []
        
        # Initialize feature importances accumulator
        feature_importances = np.zeros(n_features)
        
        # Build n_estimators trees
        for i in range(self.n_estimators):
            # Step 1: Bootstrap sampling - sample WITH replacement
            bootstrap_indices = self._rng.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            
            # Step 2: Create a DecisionTree with specified parameters
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                criterion='gini' if self.task_type_ == 'classification' else 'mse',
                max_features=self.max_features,
                random_state=None
            )
            
            # Step 3: Fit tree on bootstrap sample
            tree.fit(X_bootstrap, y_bootstrap)
            
            # Step 4: Store fitted tree
            self.estimators_.append(tree)
            
            # Accumulate feature importances
            if tree.feature_importances_ is not None:
                feature_importances += tree.feature_importances_
        
        # Step 5: Calculate average feature importances across all trees
        self.feature_importances_ = feature_importances / self.n_estimators
        
        # Step 6: Calculate OOB (Out-of-Bag) score if requested
        if self.oob_score:
            self.oob_score_ = self._calculate_oob_score(X, y)
        
        return self
    
    def _calculate_oob_score(self, X, y):
        """Calculate out-of-bag score."""
        n_samples = len(y)
        
        if self.task_type_ == 'classification':
            oob_predictions = [[] for _ in range(n_samples)]
        else:
            oob_predictions = [[] for _ in range(n_samples)]
        
        for tree_idx, tree in enumerate(self.estimators_):
            tree_pred = tree.predict(X)
            for sample_idx in range(n_samples):
                oob_predictions[sample_idx].append(tree_pred[sample_idx])
        
        # Aggregate OOB predictions
        if self.task_type_ == 'classification':
            oob_pred = np.array([
                max(set(preds), key=preds.count) if preds else 0
                for preds in oob_predictions
            ])
            return np.mean(oob_pred == y)
        else:
            oob_pred = np.array([
                np.mean(preds) if preds else 0
                for preds in oob_predictions
            ])
            ss_res = np.sum((y - oob_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    def predict(self, X):
        """Predict class labels or regression values."""
        if not self.estimators_:
            raise ValueError("Forest has not been fitted. Call fit() first.")
        
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        n_samples = len(X)
        
        # Get predictions from all trees
        tree_predictions = np.array([
            tree.predict(X) for tree in self.estimators_
        ])
        
        if self.task_type_ == 'classification':
            # Classification: Majority voting
            predictions = np.array([
                max(set(tree_predictions[:, i]), key=list(tree_predictions[:, i]).count)
                for i in range(n_samples)
            ])
        else:
            # Regression: Average predictions from all trees
            predictions = np.mean(tree_predictions, axis=0)
        
        return predictions
    
    def predict_proba(self, X):
        """Predict class probabilities (classification only)."""
        if self.task_type_ != 'classification':
            raise ValueError("predict_proba only available for classification tasks.")
        
        if not self.estimators_:
            raise ValueError("Forest has not been fitted. Call fit() first.")
        
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        n_samples = len(X)
        n_classes = len(self.classes_)
        
        # Get probabilities from all trees and average them
        proba_sum = np.zeros((n_samples, n_classes))
        
        for tree in self.estimators_:
            tree_proba = tree.predict_proba(X)
            proba_sum += tree_proba
        
        # Average probabilities across all trees
        proba = proba_sum / len(self.estimators_)
        
        return proba
    
    def score(self, X, y):
        """Calculate accuracy (classification) or R² (regression)."""
        predictions = self.predict(X)
        
        if self.task_type_ == 'classification':
            return np.mean(predictions == y)
        else:
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return (f"RandomForest(n_estimators={self.n_estimators}, "
                f"max_depth={self.max_depth}, "
                f"max_features='{self.max_features}', "
                f"random_state={self.random_state})")
