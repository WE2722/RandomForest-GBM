"""
Decision Tree Package for Random Forest and GBM.

This package provides a CART-style Decision Tree implementation designed
to be used as a reusable black-box module for Random Forest (Breiman 2001)
and Gradient Boosting Machine implementations.

TWO MODES:
    - DETERMINISTIC (GBM): max_features=None -> considers ALL features per split
    - RANDOMIZED (RF): max_features='sqrt' -> samples features per split

Classes
-------
DecisionTree
    Main decision tree classifier/regressor.
Node
    Tree node structure with EXACT 5 attributes for teammate compatibility.

Functions
---------
gini
    Gini impurity: sum(p_i * (1 - p_i))
entropy
    Entropy: -sum(p_i * log2(p_i))
mse
    Mean squared error: mean((y - mean(y))^2)

Examples
--------
>>> from decision_tree import DecisionTree
>>> import numpy as np

# Classification (GBM mode - deterministic, all features):
>>> tree = DecisionTree(max_depth=3, criterion='gini', max_features=None)
>>> tree.fit(X_train, y_train)
>>> predictions = tree.predict(X_test)

# Classification (RF mode - randomized, feature subset):
>>> tree = DecisionTree(max_depth=None, criterion='gini', 
...                     max_features='sqrt', random_state=42)
>>> tree.fit(X_train, y_train)
>>> predictions = tree.predict(X_test)

# Regression:
>>> tree = DecisionTree(criterion='mse', max_depth=5)
>>> tree.fit(X_train, y_train)
>>> predictions = tree.predict(X_test)

# Access tree structure for RF/GBM:
>>> tree.root_  # Root node
>>> tree.n_features_  # Number of features
>>> tree.classes_  # Class labels (classification)
>>> tree.feature_importances_  # Feature importances
"""

from .tree import DecisionTree
from .node import Node
from .metrics import (
    gini, 
    entropy, 
    mse, 
    gini_from_proportions,
    entropy_from_proportions,
    information_gain,
    CRITERION_FUNCTIONS
)
from .splitter import Splitter

__all__ = [
    # Main classes
    'DecisionTree',
    'Node',
    'Splitter',
    # Metric functions
    'gini',
    'entropy',
    'mse',
    'gini_from_proportions',
    'entropy_from_proportions',
    'information_gain',
    'CRITERION_FUNCTIONS'
]

__version__ = '1.0.0'
__author__ = 'Member 1'
