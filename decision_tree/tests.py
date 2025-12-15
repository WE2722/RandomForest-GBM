"""
Unit tests for Decision Tree module.

This module contains comprehensive tests for all components of the
decision tree implementation.

Run tests:
    python -m decision_tree.tests
    
Or from project root:
    python -m pytest decision_tree/tests.py -v
"""

import numpy as np
from .tree import DecisionTree
from .node import Node
from .metrics import gini, entropy, mse, information_gain
from .splitter import Splitter


def test_gini_impurity():
    """Test Gini impurity calculations."""
    print("Testing Gini impurity...")
    
    # Test with proportions
    assert abs(gini([0.5, 0.5]) - 0.5) < 1e-6, "Gini([0.5, 0.5]) should be 0.5"
    assert abs(gini([1.0, 0.0]) - 0.0) < 1e-6, "Gini([1.0, 0.0]) should be 0.0"
    assert abs(gini([0.25, 0.25, 0.25, 0.25]) - 0.75) < 1e-6, "Gini uniform 4-class"
    
    # Test with labels
    assert abs(gini(np.array([0, 0, 1, 1])) - 0.5) < 1e-6, "Gini from labels"
    assert abs(gini(np.array([0, 0, 0])) - 0.0) < 1e-6, "Gini pure node"
    
    print("   ✓ All Gini impurity tests passed")


def test_entropy():
    """Test entropy calculations."""
    print("Testing entropy...")
    
    # Test with proportions
    assert abs(entropy([0.5, 0.5]) - 1.0) < 1e-6, "Entropy([0.5, 0.5]) should be 1.0"
    assert abs(entropy([1.0, 0.0]) - 0.0) < 1e-6, "Entropy pure should be 0.0"
    
    # Test uniform distribution
    expected_entropy = np.log2(3)
    assert abs(entropy([1/3, 1/3, 1/3]) - expected_entropy) < 1e-6, "Entropy uniform 3-class"
    
    # Test with labels
    assert abs(entropy(np.array([0, 1])) - 1.0) < 1e-6, "Entropy from labels"
    
    print("   ✓ All entropy tests passed")


def test_mse():
    """Test MSE calculations."""
    print("Testing MSE...")
    
    # Test basic MSE
    test_y = np.array([1, 3, 5])
    expected_mse = np.mean((test_y - np.mean(test_y)) ** 2)
    assert abs(mse(test_y) - expected_mse) < 1e-6, "MSE calculation error"
    
    # Test constant values
    assert abs(mse(np.array([5, 5, 5])) - 0.0) < 1e-6, "MSE constant should be 0"
    
    # Test empty array
    assert abs(mse(np.array([])) - 0.0) < 1e-6, "MSE empty should be 0"
    
    print("   ✓ All MSE tests passed")


def test_information_gain():
    """Test information gain calculations."""
    print("Testing information gain...")
    
    # Perfect split
    y_parent = np.array([0, 0, 1, 1])
    y_left = np.array([0, 0])
    y_right = np.array([1, 1])
    gain = information_gain(y_parent, y_left, y_right, 'gini')
    assert abs(gain - 0.5) < 1e-6, "Perfect split should have gain 0.5"
    
    # No improvement split
    y_left_bad = np.array([0, 1])
    y_right_bad = np.array([0, 1])
    gain_bad = information_gain(y_parent, y_left_bad, y_right_bad, 'gini')
    assert abs(gain_bad - 0.0) < 1e-6, "No improvement split should have gain 0"
    
    print("   ✓ All information gain tests passed")


def test_node():
    """Test Node class."""
    print("Testing Node class...")
    
    # Test leaf node
    leaf = Node(leaf_value=1)
    assert leaf.is_leaf(), "Leaf node should return True for is_leaf()"
    assert leaf.feature_index == -1, "Leaf should have feature_index=-1"
    
    # Test decision node
    left = Node(leaf_value=0)
    right = Node(leaf_value=1)
    decision = Node(feature_index=0, threshold=5.0, left_child=left, right_child=right)
    assert not decision.is_leaf(), "Decision node should not be leaf"
    assert decision.feature_index == 0, "Feature index should be 0"
    
    print("   ✓ All Node tests passed")


def test_splitter():
    """Test Splitter class."""
    print("Testing Splitter class...")
    
    # Simple classification data
    X = np.array([[1], [2], [3], [4], [5], [6]])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    splitter = Splitter(criterion='gini', min_samples_leaf=1, task_type='classification')
    feature_idx, threshold, gain = splitter.find_best_split(X, y, np.array([0]))
    
    assert feature_idx == 0, "Should split on feature 0"
    assert 2.5 <= threshold <= 3.5, f"Threshold should be around 3, got {threshold}"
    assert gain > 0, "Gain should be positive"
    
    # Test split_data
    X_left, y_left, X_right, y_right = splitter.split_data(X, y, feature_idx, threshold)
    assert len(y_left) == 3, "Left should have 3 samples"
    assert len(y_right) == 3, "Right should have 3 samples"
    
    print("   ✓ All Splitter tests passed")


def test_classification():
    """Test classification tasks."""
    print("Testing classification...")
    
    # Simple iris-like data
    np.random.seed(42)
    X = np.array([
        [5.1, 3.5], [4.9, 3.0], [4.7, 3.2],  # Class 0
        [7.0, 3.2], [6.4, 3.2], [6.9, 3.1],  # Class 1
        [6.3, 3.3], [5.8, 2.7], [7.1, 3.0],  # Class 1
        [4.6, 3.1], [5.0, 3.6], [5.4, 3.9],  # Class 0
    ])
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0])
    
    # Test Gini criterion
    tree_gini = DecisionTree(max_depth=3, criterion='gini', random_state=42)
    tree_gini.fit(X, y)
    predictions = tree_gini.predict(X)
    accuracy = np.mean(predictions == y)
    assert accuracy >= 0.9, f"Gini accuracy should be >= 90%, got {accuracy*100}%"
    print(f"   ✓ Gini criterion accuracy: {accuracy*100:.1f}%")
    
    # Test Entropy criterion
    tree_entropy = DecisionTree(max_depth=3, criterion='entropy', random_state=42)
    tree_entropy.fit(X, y)
    predictions_ent = tree_entropy.predict(X)
    accuracy_ent = np.mean(predictions_ent == y)
    assert accuracy_ent >= 0.9, f"Entropy accuracy should be >= 90%"
    print(f"   ✓ Entropy criterion accuracy: {accuracy_ent*100:.1f}%")


def test_regression():
    """Test regression tasks."""
    print("Testing regression...")
    
    X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    y = np.array([2.1, 4.0, 6.1, 8.0, 10.2, 12.0, 13.9, 16.1, 18.0, 19.9])
    
    tree = DecisionTree(max_depth=3, criterion='mse', random_state=42)
    tree.fit(X, y)
    predictions = tree.predict(X)
    
    reg_mse = np.mean((predictions - y) ** 2)
    assert reg_mse < 2.0, f"Regression MSE should be < 2.0, got {reg_mse}"
    print(f"   ✓ Regression MSE: {reg_mse:.4f}")


def test_stopping_conditions():
    """Test stopping conditions."""
    print("Testing stopping conditions...")
    
    X = np.array([[i, i+1] for i in range(20)])
    y = np.array([0]*10 + [1]*10)
    
    # Test max_depth
    tree_depth = DecisionTree(max_depth=1, random_state=42)
    tree_depth.fit(X, y)
    assert tree_depth.get_depth() <= 1, "max_depth=1 should limit depth"
    print("   ✓ max_depth limiting works")
    
    # Test min_samples_split
    tree_split = DecisionTree(min_samples_split=100, random_state=42)
    tree_split.fit(X, y)
    assert tree_split.get_depth() == 0, "min_samples_split=100 should create leaf"
    print("   ✓ min_samples_split works")
    
    # Test min_samples_leaf
    tree_leaf = DecisionTree(min_samples_leaf=10, random_state=42)
    tree_leaf.fit(X, y)
    assert tree_leaf.get_depth() <= 1, "min_samples_leaf=10 should limit splits"
    print("   ✓ min_samples_leaf works")


def test_randomized_vs_deterministic():
    """Test randomized vs deterministic feature selection."""
    print("Testing randomized vs deterministic modes...")
    
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Deterministic mode
    tree_det = DecisionTree(max_depth=5, max_features=None, random_state=42)
    tree_det.fit(X, y)
    
    # Randomized mode
    tree_rand = DecisionTree(max_depth=5, max_features='sqrt', random_state=42)
    tree_rand.fit(X, y)
    
    acc_det = np.mean(tree_det.predict(X) == y)
    acc_rand = np.mean(tree_rand.predict(X) == y)
    
    print(f"   ✓ Deterministic mode accuracy: {acc_det*100:.1f}%")
    print(f"   ✓ Randomized mode accuracy: {acc_rand*100:.1f}%")


def test_reproducibility():
    """Test that random_state ensures reproducibility."""
    print("Testing reproducibility...")
    
    np.random.seed(42)
    X = np.random.randn(50, 5)
    y = (X[:, 0] > 0).astype(int)
    
    tree_a = DecisionTree(max_depth=3, max_features='sqrt', random_state=123)
    tree_b = DecisionTree(max_depth=3, max_features='sqrt', random_state=123)
    
    tree_a.fit(X, y)
    tree_b.fit(X, y)
    
    pred_a = tree_a.predict(X)
    pred_b = tree_b.predict(X)
    
    assert np.array_equal(pred_a, pred_b), "Same random_state should give identical results"
    print("   ✓ Reproducibility confirmed")


def test_edge_cases():
    """Test edge cases."""
    print("Testing edge cases...")
    
    # Pure node
    X_pure = np.array([[1, 2], [3, 4], [5, 6]])
    y_pure = np.array([1, 1, 1])
    tree_pure = DecisionTree()
    tree_pure.fit(X_pure, y_pure)
    assert tree_pure.get_depth() == 0, "Pure node should be leaf"
    print("   ✓ Pure node handled")
    
    # Single sample
    X_single = np.array([[1, 2]])
    y_single = np.array([0])
    tree_single = DecisionTree()
    tree_single.fit(X_single, y_single)
    assert tree_single.predict(X_single)[0] == 0, "Single sample prediction failed"
    print("   ✓ Single sample handled")
    
    # 1D input
    X_1d = np.array([1, 2, 3, 4, 5])
    y_1d = np.array([0, 0, 1, 1, 1])
    tree_1d = DecisionTree()
    tree_1d.fit(X_1d, y_1d)
    assert tree_1d.predict(np.array([1.5]))[0] == 0, "1D input handling failed"
    print("   ✓ 1D input handled")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Running Decision Tree Unit Tests")
    print("=" * 60)
    
    test_gini_impurity()
    test_entropy()
    test_mse()
    test_information_gain()
    test_node()
    test_splitter()
    test_classification()
    test_regression()
    test_stopping_conditions()
    test_randomized_vs_deterministic()
    test_reproducibility()
    test_edge_cases()
    
    print("=" * 60)
    print("All tests passed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
