"""
Node class for Decision Tree structure.

This module defines the Node class that represents individual nodes
in the decision tree. The Node structure is designed to be directly
accessible by RandomForest and GBM implementations.

EXACT ATTRIBUTES (as per specification):
    - feature_index: int (-1 for leaf)
    - threshold: float (split value)
    - left_child: Node or None
    - right_child: Node or None  
    - leaf_value: float (prediction value)
"""

import numpy as np
from typing import Union, Optional


class Node:
    """
    Node class representing a single node in the decision tree.
    
    This class has EXACTLY 5 core attributes as required by the specification
    for compatibility with RandomForest and GBM implementations.
    
    Attributes
    ----------
    feature_index : int
        Index of the splitting feature. -1 for leaf nodes.
    threshold : float
        Split threshold value for the feature.
    left_child : Node or None
        Left child node (samples where feature <= threshold).
    right_child : Node or None
        Right child node (samples where feature > threshold).
    leaf_value : float or int
        Prediction value for leaf nodes (class label for classification,
        mean value for regression).
    
    Examples
    --------
    >>> # Create a leaf node
    >>> leaf = Node(leaf_value=1)
    >>> leaf.is_leaf()
    True
    >>> leaf.feature_index
    -1
    
    >>> # Create a decision node
    >>> left = Node(leaf_value=0)
    >>> right = Node(leaf_value=1)
    >>> decision = Node(feature_index=0, threshold=5.0, 
    ...                 left_child=left, right_child=right)
    >>> decision.is_leaf()
    False
    """
    
    # Use __slots__ for memory efficiency (important for large forests)
    __slots__ = ['feature_index', 'threshold', 'left_child', 'right_child', 'leaf_value']
    
    def __init__(
        self,
        feature_index: int = -1,
        threshold: float = 0.0,
        left_child: Optional['Node'] = None,
        right_child: Optional['Node'] = None,
        leaf_value: Optional[Union[int, float]] = None
    ):
        """
        Initialize a Node.
        
        Parameters
        ----------
        feature_index : int, default=-1
            Index of the splitting feature. -1 indicates a leaf node.
        threshold : float, default=0.0
            Split threshold value.
        left_child : Node or None, default=None
            Left child node.
        right_child : Node or None, default=None
            Right child node.
        leaf_value : int or float or None, default=None
            Prediction value for leaf nodes.
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.leaf_value = leaf_value
    
    def is_leaf(self) -> bool:
        """
        Check if this node is a leaf node.
        
        A node is considered a leaf if it has no children (both left_child
        and right_child are None) OR if feature_index is -1.
        
        Returns
        -------
        bool
            True if this is a leaf node, False otherwise.
        """
        return self.feature_index == -1 or (self.left_child is None and self.right_child is None)
    
    def __repr__(self) -> str:
        """
        String representation of the node.
        
        Returns
        -------
        str
            Human-readable string representation.
        """
        if self.is_leaf():
            return f"Node(leaf_value={self.leaf_value})"
        return f"Node(feature={self.feature_index}, threshold={self.threshold:.4f})"
    
    def __str__(self) -> str:
        """
        String representation for print().
        
        Returns
        -------
        str
            Human-readable string.
        """
        return self.__repr__()
