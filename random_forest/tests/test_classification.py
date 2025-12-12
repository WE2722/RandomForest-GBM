"""
Random Forest Classification Tests with Visualization.

This module tests the Random Forest on classification tasks using various
datasets and provides visualizations of the results.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from random_forest import RandomForest


def plot_decision_boundary(model, X, y, title="Decision Boundary", ax=None):
    """
    Plot decision boundary for 2D classification.
    
    Parameters
    ----------
    model : RandomForest
        Fitted Random Forest model.
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
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
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
    """Test binary classification with Random Forest."""
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
    
    # Train Random Forest
    print("\nTraining Random Forest Classifier...")
    rf = RandomForest(
        n_estimators=10,
        max_depth=5,
        max_features='sqrt',
        random_state=42
    )
    rf.fit(X, y)
    
    # Evaluate
    predictions = rf.predict(X)
    accuracy = np.mean(predictions == y)
    
    print(f"\nRandom Forest Parameters:")
    print(f"  - Number of trees: 10")
    print(f"  - Max depth per tree: 5")
    print(f"  - Max features: sqrt(n_features) = {int(np.sqrt(X.shape[1]))}")
    print(f"\nResults:")
    print(f"  - Training accuracy: {accuracy * 100:.2f}%")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    plot_decision_boundary(rf, X, y, 
                          f"Binary Classification (Random Forest)\nAccuracy: {accuracy*100:.1f}%",
                          axes[0])
    
    # Feature importances
    importances = rf.feature_importances_
    ax = axes[1]
    ax.bar(['Feature 0', 'Feature 1'], importances)
    ax.set_ylabel('Importance')
    ax.set_title('Feature Importances')
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'images', 
                'rf_binary_classification.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ Plot saved: rf_binary_classification.png")
    
    return accuracy


def test_multiclass_classification():
    """Test multi-class classification with Random Forest."""
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
    
    # Test different number of trees
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    n_trees_list = [1, 3, 5, 10, 20, 50]
    
    for ax, n_trees in zip(axes.flat, n_trees_list):
        rf = RandomForest(
            n_estimators=n_trees,
            max_depth=5,
            max_features='sqrt',
            random_state=42
        )
        rf.fit(X, y)
        predictions = rf.predict(X)
        accuracy = np.mean(predictions == y)
        
        plot_decision_boundary(rf, X, y,
                              f"n_trees={n_trees}, Acc={accuracy*100:.1f}%",
                              ax)
    
    plt.suptitle('Multi-class Classification: Effect of n_estimators', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'images', 
                'rf_multiclass_classification.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ Plot saved: rf_multiclass_classification.png")
    
    # Final Random Forest
    rf_final = RandomForest(
        n_estimators=10,
        max_depth=5,
        max_features='sqrt',
        random_state=42
    )
    rf_final.fit(X, y)
    accuracy_final = np.mean(rf_final.predict(X) == y)
    
    print(f"\nFinal Random Forest (n_trees=10):")
    print(f"  - Training accuracy: {accuracy_final * 100:.2f}%")
    
    return accuracy_final


def test_xor_problem():
    """Test non-linear XOR problem with Random Forest."""
    print("\n" + "="*60)
    print("TEST 3: XOR Problem (Non-linear)")
    print("="*60)
    
    # Generate XOR data
    np.random.seed(42)
    n_samples = 200
    
    X = np.random.randn(n_samples, 2) * 2
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    
    # Train Random Forest
    rf = RandomForest(
        n_estimators=20,
        max_depth=8,
        max_features='sqrt',
        random_state=42
    )
    rf.fit(X, y)
    
    predictions = rf.predict(X)
    accuracy = np.mean(predictions == y)
    
    print(f"\nRandom Forest Parameters:")
    print(f"  - Number of trees: 20")
    print(f"  - Max depth: 8")
    print(f"\nResults:")
    print(f"  - Accuracy on XOR: {accuracy * 100:.2f}%")
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_decision_boundary(rf, X, y,
                          f"XOR Problem (Random Forest)\nAccuracy: {accuracy*100:.1f}%",
                          ax)
    plt.savefig(os.path.join(os.path.dirname(__file__), 'images', 
                'rf_xor_problem.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ Plot saved: rf_xor_problem.png")
    
    return accuracy


def test_feature_importances():
    """Test feature importance calculation."""
    print("\n" + "="*60)
    print("TEST 4: Feature Importances")
    print("="*60)
    
    # Generate data where features have different importances
    np.random.seed(42)
    n_samples = 300
    
    X = np.random.randn(n_samples, 5)
    # Feature 0 is most important (directly determines the class)
    y = (X[:, 0] > 0).astype(int)
    
    # Train Random Forest
    rf = RandomForest(
        n_estimators=20,
        max_depth=5,
        max_features='sqrt',
        random_state=42
    )
    rf.fit(X, y)
    
    # Get importances
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print(f"\nFeature Importances:")
    for i, idx in enumerate(indices):
        print(f"  Feature {idx}: {importances[idx]:.4f}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(importances)), importances[indices])
    ax.set_xlabel('Feature')
    ax.set_ylabel('Importance')
    ax.set_title('Random Forest Feature Importances')
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([f'F{idx}' for idx in indices])
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'images', 
                'rf_feature_importances.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ Plot saved: rf_feature_importances.png")
    
    return importances


if __name__ == '__main__':
    # Ensure images directory exists
    img_dir = os.path.join(os.path.dirname(__file__), 'images')
    os.makedirs(img_dir, exist_ok=True)
    
    test_binary_classification()
    test_multiclass_classification()
    test_xor_problem()
    test_feature_importances()
    
    print("\n" + "="*60)
    print("All Random Forest Classification Tests Completed!")
    print("="*60)
