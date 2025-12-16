"""
Regression Random Forest Tests with Visualization.

This module tests the RandomForest on regression tasks using various
datasets and provides visualizations of the results.

Author: Member 1
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forest import RandomForest


def plot_regression_1d(forest, X, y, title="Regression Random Forest", ax=None):
    """
    Plot 1D regression results.
    
    Parameters
    ----------
    forest : RandomForest
        Fitted random forest.
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
    y_pred_line = forest.predict(X_line)
    
    # Plot data points
    ax.scatter(X_sorted, y_sorted, alpha=0.6, s=50, label='Actual data')
    ax.plot(X_line, y_pred_line, 'r-', linewidth=2, label='Random Forest')
    
    # Calculate metrics
    y_pred = forest.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mse)
    
    ax.set_title(f'{title}\nMSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return mse, rmse, r2


def test_1d_sine_wave():
    """Test on 1D sine wave with noise."""
    print("\n" + "="*50)
    print("TEST 1: 1D Sine Wave Regression")
    print("="*50)
    
    # Generate data
    np.random.seed(42)
    X = np.linspace(0, 4*np.pi, 200).reshape(-1, 1)
    y = np.sin(X[:, 0]) + np.random.normal(0, 0.1, 200)
    
    # Train forest
    forest = RandomForest(n_estimators=50, max_depth=10, random_state=42)
    forest.fit(X, y)
    
    # Evaluate
    y_pred = forest.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"R²:   {r2:.6f}")
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    plot_regression_1d(forest, X, y, title="1D Sine Wave Regression", ax=ax)
    plt.tight_layout()
    plt.savefig('regression_sine_wave.png', dpi=150)
    print("\n✓ Plot saved: regression_sine_wave.png")
    plt.close()
    
    return mse, rmse, r2, mae


def test_quadratic():
    """Test on quadratic function with noise."""
    print("\n" + "="*50)
    print("TEST 2: Quadratic Function Regression")
    print("="*50)
    
    # Generate data
    np.random.seed(42)
    X = np.linspace(-5, 5, 200).reshape(-1, 1)
    y = X[:, 0]**2 + np.random.normal(0, 2, 200)
    
    # Train forest
    forest = RandomForest(n_estimators=50, max_depth=15, random_state=42)
    forest.fit(X, y)
    
    # Evaluate
    y_pred = forest.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"R²:   {r2:.6f}")
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    plot_regression_1d(forest, X, y, title="Quadratic Function Regression", ax=ax)
    plt.tight_layout()
    plt.savefig('regression_quadratic.png', dpi=150)
    print("\n✓ Plot saved: regression_quadratic.png")
    plt.close()
    
    return mse, rmse, r2, mae


def test_multi_feature_regression():
    """Test on multi-feature dataset."""
    print("\n" + "="*50)
    print("TEST 3: Multi-Feature Regression")
    print("="*50)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 300
    X = np.random.randn(n_samples, 5)
    y = 3*X[:, 0] + 2*X[:, 1]**2 - X[:, 2] + np.random.normal(0, 0.5, n_samples)
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train forest
    forest = RandomForest(n_estimators=50, max_depth=10, random_state=42)
    forest.fit(X_train, y_train)
    
    # Evaluate
    y_train_pred = forest.predict(X_train)
    y_test_pred = forest.predict(X_test)
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"Train MSE: {train_mse:.6f}, R²: {train_r2:.6f}")
    print(f"Test MSE:  {test_mse:.6f}, R²: {test_r2:.6f}")
    
    # Feature importances
    if forest.feature_importances_ is not None:
        print("\nFeature Importances:")
        for i, imp in enumerate(forest.feature_importances_):
            print(f"  Feature {i}: {imp:.6f}")
    
    return train_mse, train_r2, test_mse, test_r2


if __name__ == "__main__":
    print("\n" + "="*70)
    print("RANDOM FOREST REGRESSION TESTS")
    print("="*70)
    
    try:
        # Run tests
        results_1d_sine = test_1d_sine_wave()
        results_quadratic = test_quadratic()
        results_multi = test_multi_feature_regression()
        
        # Summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print("\n✓ All regression tests completed successfully!")
        print("✓ Generated visualizations:")
        print("  - regression_sine_wave.png")
        print("  - regression_quadratic.png")
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
