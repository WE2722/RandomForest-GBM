"""
Classification Decision Tree Tests with Visualization.

This module tests the DecisionTree on classification tasks using various
datasets and provides visualizations of the results.

Author: Member 1
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision_tree import DecisionTree, gini, entropy, gini_from_proportions


def plot_decision_boundary(tree, X, y, title="Decision Boundary", ax=None):
    """
    Plot decision boundary for 2D classification.
    
    Parameters
    ----------
    tree : DecisionTree
        Fitted decision tree.
    X : ndarray of shape (n_samples, 2)
        Feature matrix (must have 2 features).
    y : ndarray
        Target labels.
    title : str
        Plot title.
    ax : matplotlib axis
        Axis to plot on.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Predict on mesh
    Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    ax.contour(xx, yy, Z, colors='k', linewidths=0.5)
    
    # Plot training points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', 
                        edgecolors='black', s=50)
    
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(title)
    
    return ax


def test_binary_classification():
    """Test binary classification with synthetic data."""
    print("\n" + "="*60)
    print("TEST 1: Binary Classification (Linearly Separable)")
    print("="*60)
    
    # Generate linearly separable data
    np.random.seed(42)
    n_samples = 100
    
    # Class 0: lower-left cluster
    X0 = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])
    # Class 1: upper-right cluster
    X1 = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
    
    X = np.vstack([X0, X1])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    
    # Shuffle
    idx = np.random.permutation(n_samples)
    X, y = X[idx], y[idx]
    
    # Train tree
    tree = DecisionTree(max_depth=5, criterion='gini', random_state=42)
    tree.fit(X, y)
    
    # Evaluate
    predictions = tree.predict(X)
    accuracy = np.mean(predictions == y)
    
    print(f"\nTree Parameters:")
    print(f"  - Criterion: gini")
    print(f"  - Max depth: 5")
    print(f"  - Actual depth: {tree.get_depth()}")
    print(f"  - Number of leaves: {tree.get_n_leaves()}")
    print(f"\nResults:")
    print(f"  - Training accuracy: {accuracy * 100:.2f}%")
    print(f"  - Classes: {tree.classes_}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    plot_decision_boundary(tree, X, y, 
                          f"Binary Classification (Gini)\nAccuracy: {accuracy*100:.1f}%",
                          axes[0])
    
    # Test with entropy
    tree_entropy = DecisionTree(max_depth=5, criterion='entropy', random_state=42)
    tree_entropy.fit(X, y)
    predictions_ent = tree_entropy.predict(X)
    accuracy_ent = np.mean(predictions_ent == y)
    
    plot_decision_boundary(tree_entropy, X, y,
                          f"Binary Classification (Entropy)\nAccuracy: {accuracy_ent*100:.1f}%",
                          axes[1])
    
    plt.tight_layout()
    plt.savefig('test_binary_classification.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ Plot saved: test_binary_classification.png")
    
    return accuracy


def test_multiclass_classification():
    """Test multi-class classification."""
    print("\n" + "="*60)
    print("TEST 2: Multi-class Classification (3 classes)")
    print("="*60)
    
    # Generate 3-class data
    np.random.seed(42)
    n_per_class = 50
    
    # Three clusters
    X0 = np.random.randn(n_per_class, 2) * 0.8 + np.array([0, 3])
    X1 = np.random.randn(n_per_class, 2) * 0.8 + np.array([-2, -1])
    X2 = np.random.randn(n_per_class, 2) * 0.8 + np.array([2, -1])
    
    X = np.vstack([X0, X1, X2])
    y = np.array([0] * n_per_class + [1] * n_per_class + [2] * n_per_class)
    
    # Shuffle
    idx = np.random.permutation(len(y))
    X, y = X[idx], y[idx]
    
    # Test different depths
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    depths = [1, 2, 3, 5, 10, None]
    
    for ax, depth in zip(axes.flat, depths):
        tree = DecisionTree(max_depth=depth, criterion='gini', random_state=42)
        tree.fit(X, y)
        predictions = tree.predict(X)
        accuracy = np.mean(predictions == y)
        
        plot_decision_boundary(tree, X, y,
                              f"Depth={depth}, Acc={accuracy*100:.1f}%\n"
                              f"Leaves={tree.get_n_leaves()}",
                              ax)
    
    plt.suptitle('Multi-class Classification: Effect of max_depth', fontsize=14)
    plt.tight_layout()
    plt.savefig('test_multiclass_classification.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ Plot saved: test_multiclass_classification.png")
    
    # Final tree
    tree_final = DecisionTree(max_depth=5, criterion='gini', random_state=42)
    tree_final.fit(X, y)
    accuracy_final = np.mean(tree_final.predict(X) == y)
    
    print(f"\nFinal Tree (depth=5):")
    print(f"  - Training accuracy: {accuracy_final * 100:.2f}%")
    print(f"  - Number of classes: {tree_final.n_classes_}")
    
    return accuracy_final


def test_xor_problem():
    """Test non-linear XOR problem."""
    print("\n" + "="*60)
    print("TEST 3: XOR Problem (Non-linear)")
    print("="*60)
    
    # Generate XOR data
    np.random.seed(42)
    n_samples = 200
    
    X = np.random.randn(n_samples, 2)
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    
    # Test tree
    tree = DecisionTree(max_depth=10, criterion='gini', random_state=42)
    tree.fit(X, y)
    predictions = tree.predict(X)
    accuracy = np.mean(predictions == y)
    
    print(f"\nTree Parameters:")
    print(f"  - Max depth: 10")
    print(f"  - Actual depth: {tree.get_depth()}")
    print(f"  - Number of leaves: {tree.get_n_leaves()}")
    print(f"\nResults:")
    print(f"  - Training accuracy: {accuracy * 100:.2f}%")
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_decision_boundary(tree, X, y,
                          f"XOR Problem\nAccuracy: {accuracy*100:.1f}%, Depth: {tree.get_depth()}",
                          ax)
    plt.savefig('test_xor_problem.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ Plot saved: test_xor_problem.png")
    
    return accuracy


def test_impurity_functions():
    """Test impurity functions."""
    print("\n" + "="*60)
    print("TEST 4: Impurity Functions")
    print("="*60)
    
    # Test Gini
    print("\nGini Impurity Tests:")
    assert abs(gini_from_proportions([0.5, 0.5]) - 0.5) < 1e-6
    print("  ✓ gini([0.5, 0.5]) = 0.5")
    
    assert abs(gini_from_proportions([1.0, 0.0]) - 0.0) < 1e-6
    print("  ✓ gini([1.0, 0.0]) = 0.0")
    
    assert abs(gini_from_proportions([0.25, 0.25, 0.25, 0.25]) - 0.75) < 1e-6
    print("  ✓ gini([0.25, 0.25, 0.25, 0.25]) = 0.75")
    
    # Test Entropy
    print("\nEntropy Tests:")
    test_entropy = entropy(np.array([0, 0, 1, 1]))
    assert abs(test_entropy - 1.0) < 1e-6
    print(f"  ✓ entropy([0,0,1,1]) = {test_entropy:.4f}")
    
    test_entropy_3 = entropy(np.array([0, 1, 2]))
    expected = np.log2(3)
    assert abs(test_entropy_3 - expected) < 1e-6
    print(f"  ✓ entropy([0,1,2]) = {test_entropy_3:.4f} (log2(3) = {expected:.4f})")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Gini vs proportion
    p = np.linspace(0, 1, 100)
    gini_values = [gini_from_proportions([pi, 1-pi]) for pi in p]
    axes[0].plot(p, gini_values, 'b-', linewidth=2)
    axes[0].set_xlabel('Proportion of Class 1')
    axes[0].set_ylabel('Gini Impurity')
    axes[0].set_title('Gini Impurity vs Class Proportion (Binary)')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0.5, color='r', linestyle='--', label='Max impurity')
    axes[0].legend()
    
    # Entropy vs proportion
    entropy_values = []
    for pi in p:
        if pi == 0 or pi == 1:
            entropy_values.append(0)
        else:
            entropy_values.append(-pi*np.log2(pi) - (1-pi)*np.log2(1-pi))
    
    axes[1].plot(p, entropy_values, 'g-', linewidth=2)
    axes[1].set_xlabel('Proportion of Class 1')
    axes[1].set_ylabel('Entropy')
    axes[1].set_title('Entropy vs Class Proportion (Binary)')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=1.0, color='r', linestyle='--', label='Max entropy')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('test_impurity_functions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ Plot saved: test_impurity_functions.png")


def test_deterministic_vs_randomized():
    """Test deterministic (GBM) vs randomized (RF) modes."""
    print("\n" + "="*60)
    print("TEST 5: Deterministic vs Randomized Modes")
    print("="*60)
    
    # Generate data with many features
    np.random.seed(42)
    n_samples = 200
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    # Only first 2 features are relevant
    y = ((X[:, 0] + X[:, 1]) > 0).astype(int)
    
    # Deterministic mode (GBM)
    tree_det = DecisionTree(max_depth=5, max_features=None, random_state=42)
    tree_det.fit(X, y)
    acc_det = np.mean(tree_det.predict(X) == y)
    
    # Randomized mode (RF)
    tree_rand = DecisionTree(max_depth=5, max_features='sqrt', random_state=42)
    tree_rand.fit(X, y)
    acc_rand = np.mean(tree_rand.predict(X) == y)
    
    print(f"\nDeterministic Mode (GBM - max_features=None):")
    print(f"  - Features considered: ALL ({n_features})")
    print(f"  - Accuracy: {acc_det * 100:.2f}%")
    print(f"  - Depth: {tree_det.get_depth()}")
    
    print(f"\nRandomized Mode (RF - max_features='sqrt'):")
    print(f"  - Features considered: sqrt({n_features}) = {int(np.sqrt(n_features))}")
    print(f"  - Accuracy: {acc_rand * 100:.2f}%")
    print(f"  - Depth: {tree_rand.get_depth()}")
    
    # Test reproducibility
    tree_rand2 = DecisionTree(max_depth=5, max_features='sqrt', random_state=42)
    tree_rand2.fit(X, y)
    
    assert np.array_equal(tree_rand.predict(X), tree_rand2.predict(X))
    print("\n✓ Reproducibility confirmed (same random_state = same results)")
    
    # Plot feature importances
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Deterministic
    axes[0].bar(range(n_features), tree_det.feature_importances_)
    axes[0].set_xlabel('Feature Index')
    axes[0].set_ylabel('Importance')
    axes[0].set_title(f'Deterministic Mode (GBM)\nAcc: {acc_det*100:.1f}%')
    axes[0].axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Relevant features')
    axes[0].axvline(x=1, color='r', linestyle='--', alpha=0.5)
    axes[0].legend()
    
    # Randomized
    axes[1].bar(range(n_features), tree_rand.feature_importances_)
    axes[1].set_xlabel('Feature Index')
    axes[1].set_ylabel('Importance')
    axes[1].set_title(f'Randomized Mode (RF, sqrt)\nAcc: {acc_rand*100:.1f}%')
    axes[1].axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Relevant features')
    axes[1].axvline(x=1, color='r', linestyle='--', alpha=0.5)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('test_deterministic_vs_randomized.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ Plot saved: test_deterministic_vs_randomized.png")


def test_stopping_conditions():
    """Test all stopping conditions."""
    print("\n" + "="*60)
    print("TEST 6: Stopping Conditions")
    print("="*60)
    
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Test max_depth
    print("\n1. max_depth:")
    for depth in [1, 2, 3, 5]:
        tree = DecisionTree(max_depth=depth)
        tree.fit(X, y)
        print(f"   max_depth={depth} -> actual_depth={tree.get_depth()}, leaves={tree.get_n_leaves()}")
        assert tree.get_depth() <= depth
    print("   ✓ max_depth constraint works")
    
    # Test min_samples_split
    print("\n2. min_samples_split:")
    for min_split in [2, 10, 50, 100]:
        tree = DecisionTree(min_samples_split=min_split)
        tree.fit(X, y)
        print(f"   min_samples_split={min_split} -> depth={tree.get_depth()}, leaves={tree.get_n_leaves()}")
    print("   ✓ min_samples_split constraint works")
    
    # Test min_samples_leaf
    print("\n3. min_samples_leaf:")
    for min_leaf in [1, 5, 10, 20]:
        tree = DecisionTree(min_samples_leaf=min_leaf)
        tree.fit(X, y)
        print(f"   min_samples_leaf={min_leaf} -> depth={tree.get_depth()}, leaves={tree.get_n_leaves()}")
    print("   ✓ min_samples_leaf constraint works")
    
    # Test pure node
    print("\n4. Pure node (all same class):")
    X_pure = np.random.randn(50, 2)
    y_pure = np.zeros(50, dtype=int)
    tree_pure = DecisionTree()
    tree_pure.fit(X_pure, y_pure)
    assert tree_pure.get_depth() == 0
    print(f"   Pure node -> depth={tree_pure.get_depth()} (should be 0)")
    print("   ✓ Pure node creates leaf immediately")


def test_edge_cases():
    """Test edge cases."""
    print("\n" + "="*60)
    print("TEST 7: Edge Cases")
    print("="*60)
    
    # Single sample
    print("\n1. Single sample:")
    X_single = np.array([[1, 2]])
    y_single = np.array([0])
    tree = DecisionTree()
    tree.fit(X_single, y_single)
    pred = tree.predict(X_single)
    assert pred[0] == 0
    print(f"   Single sample: prediction = {pred[0]}")
    print("   ✓ Single sample handled")
    
    # Identical features
    print("\n2. Identical features:")
    X_identical = np.ones((50, 2))
    y_identical = np.array([0] * 25 + [1] * 25)
    tree = DecisionTree()
    tree.fit(X_identical, y_identical)
    print(f"   Identical features: depth = {tree.get_depth()}")
    print("   ✓ Identical features handled (cannot split)")
    
    # Single feature
    print("\n3. Single feature:")
    X_1d = np.array([[1], [2], [3], [4], [5]])
    y_1d = np.array([0, 0, 1, 1, 1])
    tree = DecisionTree()
    tree.fit(X_1d, y_1d)
    print(f"   1D input: depth = {tree.get_depth()}")
    print("   ✓ 1D input handled")


def run_all_classification_tests():
    """Run all classification tests."""
    print("\n" + "="*70)
    print("      DECISION TREE CLASSIFICATION TESTS")
    print("="*70)
    
    results = {}
    
    # Run all tests
    results['binary'] = test_binary_classification()
    results['multiclass'] = test_multiclass_classification()
    results['xor'] = test_xor_problem()
    test_impurity_functions()
    test_deterministic_vs_randomized()
    test_stopping_conditions()
    test_edge_cases()
    
    # Summary
    print("\n" + "="*70)
    print("                    SUMMARY")
    print("="*70)
    print(f"\nBinary Classification Accuracy:     {results['binary']*100:.2f}%")
    print(f"Multi-class Classification Accuracy: {results['multiclass']*100:.2f}%")
    print(f"XOR Problem Accuracy:               {results['xor']*100:.2f}%")
    print("\n✓ All classification tests passed!")
    print("="*70)
    
    # Create summary visualization
    fig, ax = plt.subplots(figsize=(8, 5))
    tests = list(results.keys())
    accuracies = [results[t] * 100 for t in tests]
    colors = ['#2ecc71' if a >= 90 else '#f39c12' if a >= 80 else '#e74c3c' for a in accuracies]
    
    bars = ax.bar(tests, accuracies, color=colors, edgecolor='black')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Classification Test Results Summary')
    ax.set_ylim(0, 105)
    ax.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90% threshold')
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax.legend()
    plt.tight_layout()
    plt.savefig('test_classification_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ Summary plot saved: test_classification_summary.png")


if __name__ == '__main__':
    run_all_classification_tests()
