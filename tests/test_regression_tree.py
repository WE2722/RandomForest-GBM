"""
Regression Decision Tree Tests with Visualization.

This module tests the DecisionTree on regression tasks using various
datasets and provides visualizations of the results.

Author: Member 1
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision_tree import DecisionTree, mse


def plot_regression_1d(tree, X, y, title="Regression Tree", ax=None):
    """
    Plot 1D regression results.
    
    Parameters
    ----------
    tree : DecisionTree
        Fitted decision tree.
    X : ndarray of shape (n_samples, 1)
        Feature matrix (1 feature).
    y : ndarray
        Target values.
    title : str
        Plot title.
    ax : matplotlib axis
        Axis to plot on.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort for plotting
    sort_idx = np.argsort(X[:, 0])
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    
    # Generate predictions for smooth line
    X_line = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 500).reshape(-1, 1)
    y_pred_line = tree.predict(X_line)
    
    # Plot data points
    ax.scatter(X_sorted[:, 0], y_sorted, c='blue', alpha=0.6, label='Training data', s=30)
    
    # Plot prediction
    ax.plot(X_line[:, 0], y_pred_line, 'r-', linewidth=2, label='Tree prediction')
    
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def test_sine_regression():
    """Test regression on sine wave."""
    print("\n" + "="*60)
    print("TEST 1: Sine Wave Regression")
    print("="*60)
    
    # Generate sine data with noise
    np.random.seed(42)
    n_samples = 200
    
    X = np.sort(np.random.uniform(0, 2 * np.pi, n_samples)).reshape(-1, 1)
    y = np.sin(X[:, 0]) + 0.2 * np.random.randn(n_samples)
    
    # Test different depths
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    depths = [1, 2, 3, 5, 10, None]
    
    results = {}
    for ax, depth in zip(axes.flat, depths):
        tree = DecisionTree(max_depth=depth, criterion='mse', random_state=42)
        tree.fit(X, y)
        
        predictions = tree.predict(X)
        mse_val = np.mean((predictions - y) ** 2)
        r2 = 1 - (np.sum((y - predictions)**2) / np.sum((y - np.mean(y))**2))
        
        results[str(depth)] = {'mse': mse_val, 'r2': r2}
        
        plot_regression_1d(tree, X, y,
                          f"Depth={depth}\nMSE={mse_val:.4f}, R²={r2:.4f}",
                          ax)
    
    plt.suptitle('Sine Wave Regression: Effect of max_depth', fontsize=14)
    plt.tight_layout()
    plt.savefig('test_sine_regression.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ Plot saved: test_sine_regression.png")
    
    # Print results
    print("\nResults by depth:")
    for depth, metrics in results.items():
        print(f"  Depth {depth}: MSE={metrics['mse']:.4f}, R²={metrics['r2']:.4f}")
    
    return results


def test_polynomial_regression():
    """Test regression on polynomial data."""
    print("\n" + "="*60)
    print("TEST 2: Polynomial Regression")
    print("="*60)
    
    # Generate polynomial data
    np.random.seed(42)
    n_samples = 150
    
    X = np.sort(np.random.uniform(-3, 3, n_samples)).reshape(-1, 1)
    # y = x^3 - 2x^2 + x + noise
    y = X[:, 0]**3 - 2*X[:, 0]**2 + X[:, 0] + np.random.randn(n_samples) * 2
    
    # Train tree
    tree = DecisionTree(max_depth=8, criterion='mse', random_state=42)
    tree.fit(X, y)
    
    predictions = tree.predict(X)
    mse_val = np.mean((predictions - y) ** 2)
    r2 = 1 - (np.sum((y - predictions)**2) / np.sum((y - np.mean(y))**2))
    
    print(f"\nTree Parameters:")
    print(f"  - Max depth: 8")
    print(f"  - Actual depth: {tree.get_depth()}")
    print(f"  - Number of leaves: {tree.get_n_leaves()}")
    print(f"\nResults:")
    print(f"  - MSE: {mse_val:.4f}")
    print(f"  - R²: {r2:.4f}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_regression_1d(tree, X, y,
                      f"Polynomial Regression (x³ - 2x² + x)\nMSE={mse_val:.4f}, R²={r2:.4f}",
                      ax)
    plt.savefig('test_polynomial_regression.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ Plot saved: test_polynomial_regression.png")
    
    return r2


def test_step_function_regression():
    """Test regression on step function."""
    print("\n" + "="*60)
    print("TEST 3: Step Function Regression")
    print("="*60)
    
    # Generate step data
    np.random.seed(42)
    n_samples = 200
    
    X = np.sort(np.random.uniform(0, 10, n_samples)).reshape(-1, 1)
    y = np.where(X[:, 0] < 3, 1, np.where(X[:, 0] < 7, 3, 5)).astype(float)
    y += np.random.randn(n_samples) * 0.3  # Add noise
    
    # Train tree
    tree = DecisionTree(max_depth=5, criterion='mse', random_state=42)
    tree.fit(X, y)
    
    predictions = tree.predict(X)
    mse_val = np.mean((predictions - y) ** 2)
    r2 = 1 - (np.sum((y - predictions)**2) / np.sum((y - np.mean(y))**2))
    
    print(f"\nStep function (steps at x=3 and x=7)")
    print(f"  - Actual depth: {tree.get_depth()}")
    print(f"  - Number of leaves: {tree.get_n_leaves()}")
    print(f"  - MSE: {mse_val:.4f}")
    print(f"  - R²: {r2:.4f}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_regression_1d(tree, X, y,
                      f"Step Function Regression\nMSE={mse_val:.4f}, R²={r2:.4f}",
                      ax)
    
    # Add true step lines
    ax.axvline(x=3, color='green', linestyle='--', alpha=0.5, label='True steps')
    ax.axvline(x=7, color='green', linestyle='--', alpha=0.5)
    ax.legend()
    
    plt.savefig('test_step_function_regression.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ Plot saved: test_step_function_regression.png")
    
    return r2


def test_multidimensional_regression():
    """Test regression with multiple features."""
    print("\n" + "="*60)
    print("TEST 4: Multi-dimensional Regression")
    print("="*60)
    
    # Generate multi-dimensional data
    np.random.seed(42)
    n_samples = 500
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    # y = 3*x1 + 2*x2 - x3 + noise (only first 3 features matter)
    y = 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.5
    
    # Split data
    train_idx = np.random.choice(n_samples, int(0.8 * n_samples), replace=False)
    test_idx = np.array([i for i in range(n_samples) if i not in train_idx])
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # Train tree
    tree = DecisionTree(max_depth=10, criterion='mse', random_state=42)
    tree.fit(X_train, y_train)
    
    # Evaluate
    train_pred = tree.predict(X_train)
    test_pred = tree.predict(X_test)
    
    train_mse = np.mean((train_pred - y_train) ** 2)
    test_mse = np.mean((test_pred - y_test) ** 2)
    
    train_r2 = 1 - (np.sum((y_train - train_pred)**2) / np.sum((y_train - np.mean(y_train))**2))
    test_r2 = 1 - (np.sum((y_test - test_pred)**2) / np.sum((y_test - np.mean(y_test))**2))
    
    print(f"\nData: {n_features} features, {n_samples} samples")
    print(f"True function: y = 3*x1 + 2*x2 - x3 + noise")
    print(f"\nTraining Results:")
    print(f"  - MSE: {train_mse:.4f}")
    print(f"  - R²: {train_r2:.4f}")
    print(f"\nTest Results:")
    print(f"  - MSE: {test_mse:.4f}")
    print(f"  - R²: {test_r2:.4f}")
    print(f"\nTree structure:")
    print(f"  - Depth: {tree.get_depth()}")
    print(f"  - Leaves: {tree.get_n_leaves()}")
    
    # Plot predictions vs actual
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training
    axes[0].scatter(y_train, train_pred, alpha=0.5)
    axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual')
    axes[0].set_ylabel('Predicted')
    axes[0].set_title(f'Training Set\nMSE={train_mse:.4f}, R²={train_r2:.4f}')
    axes[0].grid(True, alpha=0.3)
    
    # Test
    axes[1].scatter(y_test, test_pred, alpha=0.5)
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1].set_xlabel('Actual')
    axes[1].set_ylabel('Predicted')
    axes[1].set_title(f'Test Set\nMSE={test_mse:.4f}, R²={test_r2:.4f}')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Multi-dimensional Regression: Predicted vs Actual', fontsize=14)
    plt.tight_layout()
    plt.savefig('test_multidim_regression.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ Plot saved: test_multidim_regression.png")
    
    # Feature importances
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(n_features), tree.feature_importances_)
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Importance')
    ax.set_title('Feature Importances\n(True: x1=3, x2=2, x3=-1, x4=0, x5=0)')
    ax.set_xticks(range(n_features))
    ax.set_xticklabels([f'x{i+1}' for i in range(n_features)])
    
    # Highlight relevant features
    for i in [0, 1, 2]:
        ax.get_children()[i].set_color('green')
    
    plt.tight_layout()
    plt.savefig('test_feature_importances.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Plot saved: test_feature_importances.png")
    
    return test_r2


def test_mse_function():
    """Test MSE impurity function."""
    print("\n" + "="*60)
    print("TEST 5: MSE Impurity Function")
    print("="*60)
    
    # Test MSE calculation
    y1 = np.array([1, 2, 3, 4, 5])
    expected_mse = np.var(y1)  # MSE with mean prediction
    calculated_mse = mse(y1)
    
    print(f"\nTest 1: y = [1, 2, 3, 4, 5]")
    print(f"  Expected MSE (variance): {expected_mse:.4f}")
    print(f"  Calculated MSE: {calculated_mse:.4f}")
    assert abs(calculated_mse - expected_mse) < 1e-6
    print("  ✓ MSE calculation correct")
    
    # Test with constant array
    y2 = np.array([5, 5, 5, 5, 5])
    print(f"\nTest 2: y = [5, 5, 5, 5, 5]")
    print(f"  Expected MSE: 0")
    print(f"  Calculated MSE: {mse(y2):.4f}")
    assert abs(mse(y2)) < 1e-6
    print("  ✓ MSE = 0 for constant array")


def test_stopping_conditions_regression():
    """Test stopping conditions for regression."""
    print("\n" + "="*60)
    print("TEST 6: Stopping Conditions (Regression)")
    print("="*60)
    
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = X[:, 0] + X[:, 1] + np.random.randn(100) * 0.1
    
    # Test min_samples_leaf
    print("\nmin_samples_leaf effect:")
    results = []
    for min_leaf in [1, 5, 10, 20]:
        tree = DecisionTree(min_samples_leaf=min_leaf, criterion='mse')
        tree.fit(X, y)
        pred = tree.predict(X)
        mse_val = np.mean((pred - y) ** 2)
        results.append((min_leaf, tree.get_n_leaves(), mse_val))
        print(f"  min_samples_leaf={min_leaf}: leaves={tree.get_n_leaves()}, MSE={mse_val:.4f}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    min_leafs, n_leaves, mses = zip(*results)
    
    axes[0].plot(min_leafs, n_leaves, 'bo-')
    axes[0].set_xlabel('min_samples_leaf')
    axes[0].set_ylabel('Number of Leaves')
    axes[0].set_title('Effect of min_samples_leaf on Tree Size')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(min_leafs, mses, 'ro-')
    axes[1].set_xlabel('min_samples_leaf')
    axes[1].set_ylabel('MSE')
    axes[1].set_title('Effect of min_samples_leaf on MSE')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_stopping_conditions_regression.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ Plot saved: test_stopping_conditions_regression.png")


def test_overfitting():
    """Test and visualize overfitting."""
    print("\n" + "="*60)
    print("TEST 7: Overfitting Analysis")
    print("="*60)
    
    np.random.seed(42)
    n_samples = 100
    
    # Generate data
    X = np.sort(np.random.uniform(0, 2*np.pi, n_samples)).reshape(-1, 1)
    y_true = np.sin(X[:, 0])
    y = y_true + np.random.randn(n_samples) * 0.3
    
    # Split
    train_size = 70
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    
    # Test different depths
    depths = range(1, 20)
    train_mses = []
    test_mses = []
    
    for depth in depths:
        tree = DecisionTree(max_depth=depth, criterion='mse', random_state=42)
        tree.fit(X_train, y_train)
        
        train_pred = tree.predict(X_train)
        test_pred = tree.predict(X_test)
        
        train_mses.append(np.mean((train_pred - y_train) ** 2))
        test_mses.append(np.mean((test_pred - y_test) ** 2))
    
    # Find optimal depth
    optimal_depth = depths[np.argmin(test_mses)]
    
    print(f"\nOptimal depth: {optimal_depth}")
    print(f"Min test MSE: {min(test_mses):.4f}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Learning curves
    axes[0].plot(depths, train_mses, 'b-o', label='Training MSE')
    axes[0].plot(depths, test_mses, 'r-o', label='Test MSE')
    axes[0].axvline(x=optimal_depth, color='green', linestyle='--', label=f'Optimal depth={optimal_depth}')
    axes[0].set_xlabel('Max Depth')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('Overfitting Analysis: Train vs Test MSE')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Comparison: underfitting vs optimal vs overfitting
    for depth, label, color in [(2, 'Underfitting (d=2)', 'blue'),
                                 (optimal_depth, f'Optimal (d={optimal_depth})', 'green'),
                                 (15, 'Overfitting (d=15)', 'red')]:
        tree = DecisionTree(max_depth=depth, criterion='mse', random_state=42)
        tree.fit(X_train, y_train)
        
        X_line = np.linspace(0, 2*np.pi, 200).reshape(-1, 1)
        y_pred = tree.predict(X_line)
        
        axes[1].plot(X_line, y_pred, label=label, color=color, linewidth=2)
    
    axes[1].scatter(X_train, y_train, c='blue', alpha=0.3, s=20, label='Train')
    axes[1].scatter(X_test, y_test, c='red', alpha=0.3, s=20, label='Test')
    axes[1].plot(np.linspace(0, 2*np.pi, 100), np.sin(np.linspace(0, 2*np.pi, 100)), 
                'k--', alpha=0.5, label='True function')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('y')
    axes[1].set_title('Underfitting vs Optimal vs Overfitting')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_overfitting.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ Plot saved: test_overfitting.png")
    
    return optimal_depth


def run_all_regression_tests():
    """Run all regression tests."""
    print("\n" + "="*70)
    print("      DECISION TREE REGRESSION TESTS")
    print("="*70)
    
    results = {}
    
    # Run all tests
    results['sine'] = test_sine_regression()
    results['polynomial'] = test_polynomial_regression()
    results['step'] = test_step_function_regression()
    results['multidim'] = test_multidimensional_regression()
    test_mse_function()
    test_stopping_conditions_regression()
    results['optimal_depth'] = test_overfitting()
    
    # Summary
    print("\n" + "="*70)
    print("                    SUMMARY")
    print("="*70)
    print(f"\nPolynomial R²: {results['polynomial']:.4f}")
    print(f"Step Function R²: {results['step']:.4f}")
    print(f"Multi-dim Test R²: {results['multidim']:.4f}")
    print(f"Optimal depth found: {results['optimal_depth']}")
    print("\n✓ All regression tests passed!")
    print("="*70)
    
    # Create summary visualization
    fig, ax = plt.subplots(figsize=(8, 5))
    tests = ['Polynomial', 'Step Function', 'Multi-dim']
    r2_scores = [results['polynomial'], results['step'], results['multidim']]
    colors = ['#2ecc71' if r2 >= 0.8 else '#f39c12' if r2 >= 0.6 else '#e74c3c' for r2 in r2_scores]
    
    bars = ax.bar(tests, r2_scores, color=colors, edgecolor='black')
    ax.set_ylabel('R² Score')
    ax.set_title('Regression Test Results Summary')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='0.8 threshold')
    
    # Add value labels
    for bar, r2 in zip(bars, r2_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{r2:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax.legend()
    plt.tight_layout()
    plt.savefig('test_regression_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ Summary plot saved: test_regression_summary.png")


if __name__ == '__main__':
    run_all_regression_tests()
