"""
Random Forest Regression Tests with Visualization.

This module tests the Random Forest on regression tasks using various
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


def plot_regression_1d(model, X, y, title="Regression", ax=None):
    """
    Plot 1D regression results.
    
    Parameters
    ----------
    model : RandomForest
        Fitted Random Forest model.
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
    y_pred_line = model.predict(X_line)
    
    # Plot data points
    ax.scatter(X_sorted[:, 0], y_sorted, c='blue', alpha=0.6, label='Training data', s=30)
    
    # Plot prediction
    ax.plot(X_line[:, 0], y_pred_line, 'r-', linewidth=2, label='RF prediction')
    
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def test_sine_regression():
    """Test regression on sine wave with Random Forest."""
    print("\n" + "="*60)
    print("TEST 1: Sine Wave Regression")
    print("="*60)
    
    # Generate sine data with noise
    np.random.seed(42)
    n_samples = 200
    
    X = np.sort(np.random.uniform(0, 2 * np.pi, n_samples)).reshape(-1, 1)
    y = np.sin(X[:, 0]) + 0.2 * np.random.randn(n_samples)
    
    # Test different numbers of trees
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    n_estimators_list = [1, 3, 5, 10, 20, 50]
    
    results = {}
    for ax, n_est in zip(axes.flat, n_estimators_list):
        rf = RandomForest(
            n_estimators=n_est,
            max_depth=8,
            max_features='sqrt',
            random_state=42
        )
        rf.fit(X, y)
        
        predictions = rf.predict(X)
        mse_val = np.mean((predictions - y) ** 2)
        r2 = 1 - (np.sum((y - predictions)**2) / np.sum((y - np.mean(y))**2))
        
        results[str(n_est)] = {'mse': mse_val, 'r2': r2}
        
        plot_regression_1d(rf, X, y,
                          f"n_trees={n_est}\nMSE={mse_val:.4f}, R²={r2:.4f}",
                          ax)
    
    plt.suptitle('Sine Wave Regression: Effect of n_estimators', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'images', 
                'rf_sine_regression.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("\n[OK] Plot saved: rf_sine_regression.png")
    
    # Print results
    print("\nResults by n_estimators:")
    for n_est, metrics in results.items():
        print(f"  {n_est:2s} trees: MSE={metrics['mse']:.4f}, R²={metrics['r2']:.4f}")
    
    return results


def test_polynomial_regression():
    """Test regression on polynomial data."""
    print("\n" + "="*60)
    print("TEST 2: Polynomial Regression (x³ - 2x² + x)")
    print("="*60)
    
    # Generate polynomial data
    np.random.seed(42)
    n_samples = 150
    
    X = np.sort(np.random.uniform(-3, 3, n_samples)).reshape(-1, 1)
    # y = x^3 - 2x^2 + x + noise
    y = X[:, 0]**3 - 2*X[:, 0]**2 + X[:, 0] + np.random.randn(n_samples) * 2
    
    # Train Random Forest
    rf = RandomForest(
        n_estimators=20,
        max_depth=8,
        max_features='sqrt',
        random_state=42
    )
    rf.fit(X, y)
    
    predictions = rf.predict(X)
    mse_val = np.mean((predictions - y) ** 2)
    r2 = 1 - (np.sum((y - predictions)**2) / np.sum((y - np.mean(y))**2))
    
    print(f"\nRandom Forest Parameters:")
    print(f"  - Number of trees: 20")
    print(f"  - Max depth: 8")
    print(f"\nResults:")
    print(f"  - MSE: {mse_val:.4f}")
    print(f"  - R²: {r2:.4f}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_regression_1d(rf, X, y,
                      f"Polynomial Regression (x³ - 2x² + x)\nMSE={mse_val:.4f}, R²={r2:.4f}",
                      ax)
    plt.savefig(os.path.join(os.path.dirname(__file__), 'images', 
                'rf_polynomial_regression.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("\n[OK] Plot saved: rf_polynomial_regression.png")
    
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
    
    # Train Random Forest
    rf = RandomForest(
        n_estimators=15,
        max_depth=5,
        max_features='sqrt',
        random_state=42
    )
    rf.fit(X, y)
    
    predictions = rf.predict(X)
    mse_val = np.mean((predictions - y) ** 2)
    r2 = 1 - (np.sum((y - predictions)**2) / np.sum((y - np.mean(y))**2))
    
    print(f"\nStep function (steps at x=3 and x=7)")
    print(f"  - Number of trees: 15")
    print(f"  - Max depth: 5")
    print(f"  - MSE: {mse_val:.4f}")
    print(f"  - R²: {r2:.4f}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_regression_1d(rf, X, y,
                      f"Step Function Regression\nMSE={mse_val:.4f}, R²={r2:.4f}",
                      ax)
    
    # Add true step lines
    ax.axvline(x=3, color='green', linestyle='--', alpha=0.5, label='True steps')
    ax.axvline(x=7, color='green', linestyle='--', alpha=0.5)
    ax.legend()
    
    plt.savefig(os.path.join(os.path.dirname(__file__), 'images', 
                'rf_step_function_regression.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("\n[OK] Plot saved: rf_step_function_regression.png")
    
    return r2


if __name__ == '__main__':
    # Ensure images directory exists
    img_dir = os.path.join(os.path.dirname(__file__), 'images')
    os.makedirs(img_dir, exist_ok=True)
    
    test_sine_regression()
    test_polynomial_regression()
    test_step_function_regression()
    
    print("\n" + "="*60)
    print("All Random Forest Regression Tests Completed!")
    print("="*60)
