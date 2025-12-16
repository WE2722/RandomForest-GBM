"""
Classification Random Forest Tests with Visualization.

This module tests the RandomForest on classification tasks using various
datasets and provides visualizations of the results.

Author: Member 1
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forest import RandomForest


def plot_decision_boundary(forest, X, y, title="Decision Boundary", ax=None):
    """
    Plot decision boundary for 2D classification.
    
    Parameters
    ----------
    forest : RandomForest
        Fitted random forest.
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
    Z = forest.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and margins
    ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    
    # Plot points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', 
                        edgecolors='k', s=50, alpha=0.8)
    
    # Calculate accuracy
    y_pred = forest.predict(X)
    accuracy = accuracy_score(y, y_pred)
    
    ax.set_title(f'{title}\nAccuracy: {accuracy:.4f}')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    
    return accuracy


def test_moons():
    """Test on two moons dataset."""
    print("\n" + "="*50)
    print("TEST 1: Two Moons Dataset")
    print("="*50)
    
    # Generate data
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=300, noise=0.15, random_state=42)
    
    # Split data
    n_train = int(0.7 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    # Train forest
    forest = RandomForest(n_estimators=100, max_depth=5, random_state=42)
    forest.fit(X_train, y_train)
    
    # Evaluate
    y_train_pred = forest.predict(X_train)
    y_test_pred = forest.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    
    # Plot decision boundary
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_decision_boundary(forest, X_train, y_train, 
                          title="Two Moons - Train", ax=axes[0])
    plot_decision_boundary(forest, X_test, y_test, 
                          title="Two Moons - Test", ax=axes[1])
    plt.tight_layout()
    plt.savefig('classification_moons.png', dpi=150)
    print("✓ Plot saved: classification_moons.png")
    plt.close()
    
    return train_acc, test_acc


def test_circles():
    """Test on two circles dataset."""
    print("\n" + "="*50)
    print("TEST 2: Two Circles Dataset")
    print("="*50)
    
    # Generate data
    from sklearn.datasets import make_circles
    X, y = make_circles(n_samples=300, noise=0.1, factor=0.3, random_state=42)
    
    # Split data
    n_train = int(0.7 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    # Train forest
    forest = RandomForest(n_estimators=100, max_depth=10, random_state=42)
    forest.fit(X_train, y_train)
    
    # Evaluate
    y_train_pred = forest.predict(X_train)
    y_test_pred = forest.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    
    # Plot decision boundary
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_decision_boundary(forest, X_train, y_train, 
                          title="Two Circles - Train", ax=axes[0])
    plot_decision_boundary(forest, X_test, y_test, 
                          title="Two Circles - Test", ax=axes[1])
    plt.tight_layout()
    plt.savefig('classification_circles.png', dpi=150)
    print("✓ Plot saved: classification_circles.png")
    plt.close()
    
    return train_acc, test_acc


def test_iris():
    """Test on iris dataset."""
    print("\n" + "="*50)
    print("TEST 3: Iris Dataset")
    print("="*50)
    
    # Load data
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    
    iris = load_iris()
    X, y = iris.data, iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train forest
    forest = RandomForest(n_estimators=100, max_depth=8, random_state=42)
    forest.fit(X_train, y_train)
    
    # Evaluate
    y_train_pred = forest.predict(X_train)
    y_test_pred = forest.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    # Detailed metrics
    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"F1 Score:       {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\nConfusion Matrix:\n{cm}")
    
    # Feature importances
    if forest.feature_importances_ is not None:
        print("\nFeature Importances:")
        for i, imp in enumerate(forest.feature_importances_):
            print(f"  {iris.feature_names[i]}: {imp:.6f}")
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Iris Dataset - Confusion Matrix\nAccuracy: {test_acc:.4f}')
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    
    # Add text annotations
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='white')
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig('classification_iris_cm.png', dpi=150)
    print("\n✓ Plot saved: classification_iris_cm.png")
    plt.close()
    
    return train_acc, test_acc, precision, recall, f1


def test_multi_class():
    """Test on synthetic multi-class dataset."""
    print("\n" + "="*50)
    print("TEST 4: Synthetic Multi-Class Dataset")
    print("="*50)
    
    # Generate data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=8,
        n_redundant=2, n_classes=3, n_clusters_per_class=1,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train forest
    forest = RandomForest(n_estimators=100, max_depth=10, random_state=42)
    forest.fit(X_train, y_train)
    
    # Evaluate
    y_train_pred = forest.predict(X_train)
    y_test_pred = forest.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"F1 Score:       {f1:.4f}")
    
    return train_acc, test_acc, precision, recall, f1


if __name__ == "__main__":
    print("\n" + "="*70)
    print("RANDOM FOREST CLASSIFICATION TESTS")
    print("="*70)
    
    try:
        # Run tests
        results_moons = test_moons()
        results_circles = test_circles()
        results_iris = test_iris()
        results_multi = test_multi_class()
        
        # Summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print("\n✓ All classification tests completed successfully!")
        print("✓ Generated visualizations:")
        print("  - classification_moons.png")
        print("  - classification_circles.png")
        print("  - classification_iris_cm.png")
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
