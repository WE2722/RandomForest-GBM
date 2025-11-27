"""
Random Forest Implementation (Member 2)

Based on Breiman (2001): "Random Forests"

TODO: Implement this class!
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
        
        # TODO: Initialize these in fit()
        self.estimators_ = []
        self.feature_importances_ = None
        self.oob_score_ = None
        self.n_features_ = None
        self.n_classes_ = None
        self.classes_ = None
        self.task_type_ = None
    
    def fit(self, X, y):
        """
        Build a forest of trees from training data.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training features.
        
        y : ndarray of shape (n_samples,)
            Target values.
        
        Returns
        -------
        self : RandomForest
            Fitted estimator.
        """
        # TODO: Implement this method
        # 
        # Steps:
        # 1. Detect task type (classification vs regression)
        # 2. Initialize random state
        # 3. For each tree:
        #    a. Generate bootstrap sample (sample WITH replacement)
        #    b. Create DecisionTree with max_features='sqrt'
        #    c. Fit tree on bootstrap sample
        #    d. Store tree in self.estimators_
        # 4. Calculate feature_importances_ (average across trees)
        # 5. Calculate OOB score if requested
        #
        # HINT: Use this for bootstrap sampling:
        # bootstrap_indices = rng.choice(n_samples, size=n_samples, replace=True)
        
        raise NotImplementedError("TODO: Implement fit() method")
    
    def predict(self, X):
        """
        Predict class labels or regression values.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples to predict.
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        # TODO: Implement this method
        #
        # For classification: majority vote across all trees
        # For regression: average across all trees
        
        raise NotImplementedError("TODO: Implement predict() method")
    
    def predict_proba(self, X):
        """
        Predict class probabilities (classification only).
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples to predict.
        
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        # TODO: Implement this method
        #
        # Average predict_proba() across all trees
        
        raise NotImplementedError("TODO: Implement predict_proba() method")
    
    def _compute_oob_score(self, X, y):
        """
        Compute out-of-bag score.
        
        TODO: Implement this method (optional but recommended)
        """
        raise NotImplementedError("TODO: Implement OOB score calculation")
