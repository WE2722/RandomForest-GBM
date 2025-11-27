"""
Gradient Boosting Machine Implementation (Member 3)

Based on Friedman (2001): "Greedy Function Approximation: A Gradient Boosting Machine"

TODO: Implement this class!
"""

import numpy as np
import sys
sys.path.insert(0, '..')
from decision_tree import DecisionTree


class GradientBoostingMachine:
    """
    Gradient Boosting Machine for Regression and Classification.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting stages (trees).
    
    learning_rate : float, default=0.1
        Shrinkage parameter. Scales the contribution of each tree.
        Lower values require more trees but often achieve better generalization.
    
    max_depth : int, default=3
        Maximum depth of each tree. GBM typically uses shallow trees (3-8).
    
    min_samples_split : int, default=2
        Minimum samples required to split a node.
    
    min_samples_leaf : int, default=1
        Minimum samples required in a leaf node.
    
    subsample : float, default=1.0
        Fraction of samples used for fitting each tree.
        Setting < 1.0 enables Stochastic Gradient Boosting.
    
    loss : str, default='squared_error'
        Loss function to optimize:
        - 'squared_error': L2 loss (regression)
        - 'absolute_error': L1 loss (regression)
        - 'log_loss': Logistic loss (classification)
    
    random_state : int or None, default=None
        Random seed for reproducibility.
    
    Attributes
    ----------
    estimators_ : list of DecisionTree
        The collection of fitted trees.
    
    init_ : float
        Initial prediction (F_0).
    
    feature_importances_ : ndarray of shape (n_features,)
        Feature importances (summed across trees).
    
    train_score_ : list of float
        Training loss at each stage.
    
    n_features_ : int
        Number of features.
    """
    
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        subsample=1.0,
        loss='squared_error',
        random_state=None
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.loss = loss
        self.random_state = random_state
        
        # TODO: Initialize these in fit()
        self.estimators_ = []
        self.init_ = None
        self.feature_importances_ = None
        self.train_score_ = []
        self.n_features_ = None
    
    def fit(self, X, y):
        """
        Fit the gradient boosting model.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training features.
        
        y : ndarray of shape (n_samples,)
            Target values.
        
        Returns
        -------
        self : GradientBoostingMachine
            Fitted estimator.
        """
        # TODO: Implement this method
        #
        # Steps:
        # 1. Initialize F_0 (e.g., mean of y for squared_error)
        # 2. Initialize random state
        # 3. For each boosting stage m:
        #    a. Compute pseudo-residuals (negative gradient of loss)
        #    b. Subsample if subsample < 1.0
        #    c. Fit regression tree to residuals with max_features=None
        #    d. Update predictions: F_m = F_{m-1} + learning_rate * tree.predict(X)
        #    e. Store tree and optionally compute training loss
        # 4. Compute feature_importances_
        #
        # IMPORTANT: Use max_features=None (DETERMINISTIC mode) for GBM!
        #
        # For squared_error loss:
        #   residuals = y - current_predictions
        #
        # For absolute_error loss:
        #   residuals = np.sign(y - current_predictions)
        
        raise NotImplementedError("TODO: Implement fit() method")
    
    def predict(self, X):
        """
        Predict target values.
        
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
        # y_pred = init_ + learning_rate * sum(tree.predict(X) for tree in estimators_)
        
        raise NotImplementedError("TODO: Implement predict() method")
    
    def staged_predict(self, X):
        """
        Predict at each boosting stage (for learning curves).
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples to predict.
        
        Yields
        ------
        y_pred : ndarray of shape (n_samples,)
            Predictions at each stage.
        """
        # TODO: Implement this method
        #
        # Useful for plotting learning curves:
        # for i, y_pred in enumerate(gbm.staged_predict(X_test)):
        #     test_errors.append(mean_squared_error(y_test, y_pred))
        
        raise NotImplementedError("TODO: Implement staged_predict() method")
    
    def _compute_residuals(self, y, y_pred):
        """
        Compute pseudo-residuals (negative gradient of loss).
        
        TODO: Implement for different loss functions
        """
        if self.loss == 'squared_error':
            return y - y_pred
        elif self.loss == 'absolute_error':
            return np.sign(y - y_pred)
        else:
            raise ValueError(f"Unknown loss: {self.loss}")
    
    def _init_prediction(self, y):
        """
        Compute initial prediction F_0.
        
        TODO: Implement for different loss functions
        """
        if self.loss == 'squared_error':
            return np.mean(y)
        elif self.loss == 'absolute_error':
            return np.median(y)
        else:
            raise ValueError(f"Unknown loss: {self.loss}")
