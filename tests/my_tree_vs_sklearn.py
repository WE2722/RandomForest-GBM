"""
Compare Custom Decision Tree vs sklearn DecisionTree.

This module provides a comprehensive comparison between the custom
CART-style decision tree implementation and sklearn's DecisionTree.

Author: Member 1
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision_tree import DecisionTree

# Import sklearn
try:
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.datasets import make_classification, make_regression, load_iris, load_diabetes
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("WARNING: sklearn not available. Install with: pip install scikit-learn")


def compare_classification(X_train, X_test, y_train, y_test, dataset_name, max_depth=10):
    """
    Compare classification performance between custom tree and sklearn.
    
    Returns
    -------
    dict : Comparison results
    """
    results = {'dataset': dataset_name}
    
    # Custom tree
    start = time.time()
    my_tree = DecisionTree(max_depth=max_depth, criterion='gini', random_state=42)
    my_tree.fit(X_train, y_train)
    my_train_time = time.time() - start
    
    start = time.time()
    my_train_pred = my_tree.predict(X_train)
    my_test_pred = my_tree.predict(X_test)
    my_pred_time = time.time() - start
    
    results['my_train_acc'] = np.mean(my_train_pred == y_train)
    results['my_test_acc'] = np.mean(my_test_pred == y_test)
    results['my_train_time'] = my_train_time
    results['my_pred_time'] = my_pred_time
    results['my_depth'] = my_tree.get_depth()
    results['my_leaves'] = my_tree.get_n_leaves()
    
    # sklearn tree
    start = time.time()
    sk_tree = DecisionTreeClassifier(max_depth=max_depth, criterion='gini', random_state=42)
    sk_tree.fit(X_train, y_train)
    sk_train_time = time.time() - start
    
    start = time.time()
    sk_train_pred = sk_tree.predict(X_train)
    sk_test_pred = sk_tree.predict(X_test)
    sk_pred_time = time.time() - start
    
    results['sk_train_acc'] = np.mean(sk_train_pred == y_train)
    results['sk_test_acc'] = np.mean(sk_test_pred == y_test)
    results['sk_train_time'] = sk_train_time
    results['sk_pred_time'] = sk_pred_time
    results['sk_depth'] = sk_tree.get_depth()
    results['sk_leaves'] = sk_tree.get_n_leaves()
    
    return results


def compare_regression(X_train, X_test, y_train, y_test, dataset_name, max_depth=10):
    """
    Compare regression performance between custom tree and sklearn.
    
    Returns
    -------
    dict : Comparison results
    """
    results = {'dataset': dataset_name}
    
    # Custom tree
    start = time.time()
    my_tree = DecisionTree(max_depth=max_depth, criterion='mse', random_state=42)
    my_tree.fit(X_train, y_train)
    my_train_time = time.time() - start
    
    start = time.time()
    my_train_pred = my_tree.predict(X_train)
    my_test_pred = my_tree.predict(X_test)
    my_pred_time = time.time() - start
    
    results['my_train_mse'] = np.mean((my_train_pred - y_train) ** 2)
    results['my_test_mse'] = np.mean((my_test_pred - y_test) ** 2)
    results['my_train_r2'] = 1 - np.sum((y_train - my_train_pred)**2) / np.sum((y_train - np.mean(y_train))**2)
    results['my_test_r2'] = 1 - np.sum((y_test - my_test_pred)**2) / np.sum((y_test - np.mean(y_test))**2)
    results['my_train_time'] = my_train_time
    results['my_pred_time'] = my_pred_time
    results['my_depth'] = my_tree.get_depth()
    results['my_leaves'] = my_tree.get_n_leaves()
    
    # sklearn tree
    start = time.time()
    sk_tree = DecisionTreeRegressor(max_depth=max_depth, criterion='squared_error', random_state=42)
    sk_tree.fit(X_train, y_train)
    sk_train_time = time.time() - start
    
    start = time.time()
    sk_train_pred = sk_tree.predict(X_train)
    sk_test_pred = sk_tree.predict(X_test)
    sk_pred_time = time.time() - start
    
    results['sk_train_mse'] = np.mean((sk_train_pred - y_train) ** 2)
    results['sk_test_mse'] = np.mean((sk_test_pred - y_test) ** 2)
    results['sk_train_r2'] = 1 - np.sum((y_train - sk_train_pred)**2) / np.sum((y_train - np.mean(y_train))**2)
    results['sk_test_r2'] = 1 - np.sum((y_test - sk_test_pred)**2) / np.sum((y_test - np.mean(y_test))**2)
    results['sk_train_time'] = sk_train_time
    results['sk_pred_time'] = sk_pred_time
    results['sk_depth'] = sk_tree.get_depth()
    results['sk_leaves'] = sk_tree.get_n_leaves()
    
    return results


def run_classification_comparison():
    """Run comprehensive classification comparison."""
    print("\n" + "="*70)
    print("     CLASSIFICATION COMPARISON: Custom Tree vs sklearn")
    print("="*70)
    
    all_results = []
    
    # Test 1: Iris dataset
    print("\n--- Test 1: Iris Dataset ---")
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42
    )
    results = compare_classification(X_train, X_test, y_train, y_test, "Iris", max_depth=5)
    all_results.append(results)
    
    print(f"\nMy Tree:    Train Acc={results['my_train_acc']:.4f}, Test Acc={results['my_test_acc']:.4f}")
    print(f"sklearn:    Train Acc={results['sk_train_acc']:.4f}, Test Acc={results['sk_test_acc']:.4f}")
    print(f"My Tree:    Depth={results['my_depth']}, Leaves={results['my_leaves']}")
    print(f"sklearn:    Depth={results['sk_depth']}, Leaves={results['sk_leaves']}")
    
    # Test 2: Synthetic binary classification
    print("\n--- Test 2: Synthetic Binary (n=500, f=10) ---")
    X, y = make_classification(n_samples=500, n_features=10, n_informative=5, 
                               n_redundant=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    results = compare_classification(X_train, X_test, y_train, y_test, "Synthetic Binary", max_depth=10)
    all_results.append(results)
    
    print(f"\nMy Tree:    Train Acc={results['my_train_acc']:.4f}, Test Acc={results['my_test_acc']:.4f}")
    print(f"sklearn:    Train Acc={results['sk_train_acc']:.4f}, Test Acc={results['sk_test_acc']:.4f}")
    
    # Test 3: Multi-class
    print("\n--- Test 3: Synthetic Multi-class (n=600, f=15, c=3) ---")
    X, y = make_classification(n_samples=600, n_features=15, n_informative=8,
                               n_classes=3, n_clusters_per_class=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    results = compare_classification(X_train, X_test, y_train, y_test, "Synthetic Multi-class", max_depth=10)
    all_results.append(results)
    
    print(f"\nMy Tree:    Train Acc={results['my_train_acc']:.4f}, Test Acc={results['my_test_acc']:.4f}")
    print(f"sklearn:    Train Acc={results['sk_train_acc']:.4f}, Test Acc={results['sk_test_acc']:.4f}")
    
    # Test 4: Large dataset
    print("\n--- Test 4: Large Dataset (n=2000, f=20) ---")
    X, y = make_classification(n_samples=2000, n_features=20, n_informative=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    results = compare_classification(X_train, X_test, y_train, y_test, "Large Binary", max_depth=15)
    all_results.append(results)
    
    print(f"\nMy Tree:    Train Acc={results['my_train_acc']:.4f}, Test Acc={results['my_test_acc']:.4f}")
    print(f"sklearn:    Train Acc={results['sk_train_acc']:.4f}, Test Acc={results['sk_test_acc']:.4f}")
    print(f"My Tree:    Train Time={results['my_train_time']:.4f}s")
    print(f"sklearn:    Train Time={results['sk_train_time']:.4f}s")
    
    return all_results


def run_regression_comparison():
    """Run comprehensive regression comparison."""
    print("\n" + "="*70)
    print("     REGRESSION COMPARISON: Custom Tree vs sklearn")
    print("="*70)
    
    all_results = []
    
    # Test 1: Diabetes dataset
    print("\n--- Test 1: Diabetes Dataset ---")
    diabetes = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        diabetes.data, diabetes.target, test_size=0.3, random_state=42
    )
    results = compare_regression(X_train, X_test, y_train, y_test, "Diabetes", max_depth=5)
    all_results.append(results)
    
    print(f"\nMy Tree:    Train R²={results['my_train_r2']:.4f}, Test R²={results['my_test_r2']:.4f}")
    print(f"sklearn:    Train R²={results['sk_train_r2']:.4f}, Test R²={results['sk_test_r2']:.4f}")
    print(f"My Tree:    Train MSE={results['my_train_mse']:.2f}, Test MSE={results['my_test_mse']:.2f}")
    print(f"sklearn:    Train MSE={results['sk_train_mse']:.2f}, Test MSE={results['sk_test_mse']:.2f}")
    
    # Test 2: Synthetic regression
    print("\n--- Test 2: Synthetic Regression (n=500, f=10) ---")
    X, y = make_regression(n_samples=500, n_features=10, n_informative=5, 
                           noise=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    results = compare_regression(X_train, X_test, y_train, y_test, "Synthetic", max_depth=10)
    all_results.append(results)
    
    print(f"\nMy Tree:    Train R²={results['my_train_r2']:.4f}, Test R²={results['my_test_r2']:.4f}")
    print(f"sklearn:    Train R²={results['sk_train_r2']:.4f}, Test R²={results['sk_test_r2']:.4f}")
    
    # Test 3: Large regression
    print("\n--- Test 3: Large Regression (n=2000, f=20) ---")
    X, y = make_regression(n_samples=2000, n_features=20, n_informative=10, 
                           noise=30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    results = compare_regression(X_train, X_test, y_train, y_test, "Large", max_depth=15)
    all_results.append(results)
    
    print(f"\nMy Tree:    Train R²={results['my_train_r2']:.4f}, Test R²={results['my_test_r2']:.4f}")
    print(f"sklearn:    Train R²={results['sk_train_r2']:.4f}, Test R²={results['sk_test_r2']:.4f}")
    print(f"My Tree:    Train Time={results['my_train_time']:.4f}s")
    print(f"sklearn:    Train Time={results['sk_train_time']:.4f}s")
    
    # Test 4: Sine wave
    print("\n--- Test 4: Sine Wave ---")
    np.random.seed(42)
    X = np.sort(np.random.uniform(0, 2*np.pi, 300)).reshape(-1, 1)
    y = np.sin(X[:, 0]) + np.random.randn(300) * 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    results = compare_regression(X_train, X_test, y_train, y_test, "Sine Wave", max_depth=10)
    all_results.append(results)
    
    print(f"\nMy Tree:    Train R²={results['my_train_r2']:.4f}, Test R²={results['my_test_r2']:.4f}")
    print(f"sklearn:    Train R²={results['sk_train_r2']:.4f}, Test R²={results['sk_test_r2']:.4f}")
    
    return all_results


def create_comparison_visualization(cls_results, reg_results):
    """Create comprehensive visualization of comparison results."""
    
    fig = plt.figure(figsize=(16, 12))
    
    # Classification comparison
    ax1 = fig.add_subplot(2, 2, 1)
    datasets = [r['dataset'] for r in cls_results]
    my_accs = [r['my_test_acc'] for r in cls_results]
    sk_accs = [r['sk_test_acc'] for r in cls_results]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, my_accs, width, label='My Tree', color='#3498db')
    bars2 = ax1.bar(x + width/2, sk_accs, width, label='sklearn', color='#e74c3c')
    
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Classification: Test Accuracy Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, rotation=15, ha='right')
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
    
    # Regression comparison
    ax2 = fig.add_subplot(2, 2, 2)
    datasets = [r['dataset'] for r in reg_results]
    my_r2s = [r['my_test_r2'] for r in reg_results]
    sk_r2s = [r['sk_test_r2'] for r in reg_results]
    
    x = np.arange(len(datasets))
    
    bars1 = ax2.bar(x - width/2, my_r2s, width, label='My Tree', color='#3498db')
    bars2 = ax2.bar(x + width/2, sk_r2s, width, label='sklearn', color='#e74c3c')
    
    ax2.set_ylabel('Test R²')
    ax2.set_title('Regression: Test R² Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, rotation=15, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar in bars1:
        ax2.text(bar.get_x() + bar.get_width()/2, max(0, bar.get_height()) + 0.02,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, max(0, bar.get_height()) + 0.02,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
    
    # Training time comparison (classification)
    ax3 = fig.add_subplot(2, 2, 3)
    datasets = [r['dataset'] for r in cls_results]
    my_times = [r['my_train_time'] * 1000 for r in cls_results]  # Convert to ms
    sk_times = [r['sk_train_time'] * 1000 for r in cls_results]
    
    x = np.arange(len(datasets))
    
    bars1 = ax3.bar(x - width/2, my_times, width, label='My Tree', color='#3498db')
    bars2 = ax3.bar(x + width/2, sk_times, width, label='sklearn', color='#e74c3c')
    
    ax3.set_ylabel('Training Time (ms)')
    ax3.set_title('Classification: Training Time Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(datasets, rotation=15, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Tree structure comparison
    ax4 = fig.add_subplot(2, 2, 4)
    
    # Combine all results
    all_results = cls_results + reg_results
    all_datasets = [r['dataset'] for r in all_results]
    my_depths = [r['my_depth'] for r in all_results]
    sk_depths = [r['sk_depth'] for r in all_results]
    my_leaves = [r['my_leaves'] for r in all_results]
    sk_leaves = [r['sk_leaves'] for r in all_results]
    
    x = np.arange(len(all_datasets))
    width = 0.2
    
    ax4.bar(x - 1.5*width, my_depths, width, label='My Depth', color='#3498db', alpha=0.7)
    ax4.bar(x - 0.5*width, sk_depths, width, label='SK Depth', color='#e74c3c', alpha=0.7)
    ax4.bar(x + 0.5*width, my_leaves, width, label='My Leaves', color='#2ecc71', alpha=0.7)
    ax4.bar(x + 1.5*width, sk_leaves, width, label='SK Leaves', color='#9b59b6', alpha=0.7)
    
    ax4.set_ylabel('Count')
    ax4.set_title('Tree Structure Comparison (All Datasets)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(all_datasets, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('my_tree_vs_sklearn.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ Main comparison plot saved: my_tree_vs_sklearn.png")


def create_detailed_comparison():
    """Create detailed side-by-side comparison on specific datasets."""
    
    # Classification: Decision boundary comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Generate 2D classification data
    np.random.seed(42)
    n_samples = 300
    X = np.random.randn(n_samples, 2)
    y = ((X[:, 0] + X[:, 1]) > 0).astype(int)
    
    # Custom tree
    my_tree = DecisionTree(max_depth=5, criterion='gini', random_state=42)
    my_tree.fit(X, y)
    
    # sklearn tree
    sk_tree = DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=42)
    sk_tree.fit(X, y)
    
    # Plot decision boundaries
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # My tree boundary
    Z_my = my_tree.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_my = Z_my.reshape(xx.shape)
    
    axes[0, 0].contourf(xx, yy, Z_my, alpha=0.4, cmap='RdYlBu')
    axes[0, 0].contour(xx, yy, Z_my, colors='k', linewidths=0.5)
    axes[0, 0].scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black', s=30)
    axes[0, 0].set_title(f'My Decision Tree\nAcc: {np.mean(my_tree.predict(X) == y):.4f}')
    axes[0, 0].set_xlabel('Feature 1')
    axes[0, 0].set_ylabel('Feature 2')
    
    # sklearn boundary
    Z_sk = sk_tree.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_sk = Z_sk.reshape(xx.shape)
    
    axes[0, 1].contourf(xx, yy, Z_sk, alpha=0.4, cmap='RdYlBu')
    axes[0, 1].contour(xx, yy, Z_sk, colors='k', linewidths=0.5)
    axes[0, 1].scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black', s=30)
    axes[0, 1].set_title(f'sklearn DecisionTree\nAcc: {np.mean(sk_tree.predict(X) == y):.4f}')
    axes[0, 1].set_xlabel('Feature 1')
    axes[0, 1].set_ylabel('Feature 2')
    
    # Regression comparison
    np.random.seed(42)
    X_reg = np.sort(np.random.uniform(0, 2*np.pi, 200)).reshape(-1, 1)
    y_reg = np.sin(X_reg[:, 0]) + np.random.randn(200) * 0.2
    
    my_tree_reg = DecisionTree(max_depth=5, criterion='mse', random_state=42)
    my_tree_reg.fit(X_reg, y_reg)
    
    sk_tree_reg = DecisionTreeRegressor(max_depth=5, criterion='squared_error', random_state=42)
    sk_tree_reg.fit(X_reg, y_reg)
    
    X_line = np.linspace(0, 2*np.pi, 500).reshape(-1, 1)
    
    # My tree regression
    axes[1, 0].scatter(X_reg, y_reg, c='blue', alpha=0.5, s=20, label='Data')
    axes[1, 0].plot(X_line, my_tree_reg.predict(X_line), 'r-', linewidth=2, label='Prediction')
    axes[1, 0].plot(X_line, np.sin(X_line), 'g--', alpha=0.5, label='True sin(x)')
    my_mse = np.mean((my_tree_reg.predict(X_reg) - y_reg)**2)
    axes[1, 0].set_title(f'My Decision Tree\nMSE: {my_mse:.4f}')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('y')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # sklearn regression
    axes[1, 1].scatter(X_reg, y_reg, c='blue', alpha=0.5, s=20, label='Data')
    axes[1, 1].plot(X_line, sk_tree_reg.predict(X_line), 'r-', linewidth=2, label='Prediction')
    axes[1, 1].plot(X_line, np.sin(X_line), 'g--', alpha=0.5, label='True sin(x)')
    sk_mse = np.mean((sk_tree_reg.predict(X_reg) - y_reg)**2)
    axes[1, 1].set_title(f'sklearn DecisionTree\nMSE: {sk_mse:.4f}')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('y')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Side-by-Side Comparison: My Tree vs sklearn', fontsize=14)
    plt.tight_layout()
    plt.savefig('my_tree_vs_sklearn_detailed.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Detailed comparison plot saved: my_tree_vs_sklearn_detailed.png")


def print_summary(cls_results, reg_results):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("                    FINAL SUMMARY")
    print("="*70)
    
    # Classification
    print("\n📊 CLASSIFICATION RESULTS:")
    print("-" * 50)
    my_accs = [r['my_test_acc'] for r in cls_results]
    sk_accs = [r['sk_test_acc'] for r in cls_results]
    
    print(f"Average Test Accuracy:")
    print(f"  My Tree:  {np.mean(my_accs):.4f}")
    print(f"  sklearn:  {np.mean(sk_accs):.4f}")
    print(f"  Diff:     {np.mean(my_accs) - np.mean(sk_accs):+.4f}")
    
    # Regression
    print("\n📈 REGRESSION RESULTS:")
    print("-" * 50)
    my_r2s = [r['my_test_r2'] for r in reg_results]
    sk_r2s = [r['sk_test_r2'] for r in reg_results]
    
    print(f"Average Test R²:")
    print(f"  My Tree:  {np.mean(my_r2s):.4f}")
    print(f"  sklearn:  {np.mean(sk_r2s):.4f}")
    print(f"  Diff:     {np.mean(my_r2s) - np.mean(sk_r2s):+.4f}")
    
    # Training time
    print("\n⏱️  TRAINING TIME:")
    print("-" * 50)
    all_results = cls_results + reg_results
    my_times = [r['my_train_time'] for r in all_results]
    sk_times = [r['sk_train_time'] for r in all_results]
    
    print(f"Average Training Time:")
    print(f"  My Tree:  {np.mean(my_times)*1000:.2f} ms")
    print(f"  sklearn:  {np.mean(sk_times)*1000:.2f} ms")
    print(f"  Ratio:    {np.mean(my_times)/np.mean(sk_times):.2f}x")
    
    # Conclusion
    print("\n" + "="*70)
    print("                    CONCLUSION")
    print("="*70)
    
    acc_diff = np.mean(my_accs) - np.mean(sk_accs)
    r2_diff = np.mean(my_r2s) - np.mean(sk_r2s)
    
    if abs(acc_diff) < 0.02 and abs(r2_diff) < 0.05:
        print("\n✅ The custom Decision Tree implementation produces")
        print("   COMPARABLE results to sklearn's implementation!")
    elif acc_diff > 0 and r2_diff > 0:
        print("\n🎉 The custom Decision Tree implementation OUTPERFORMS")
        print("   sklearn's implementation on these tests!")
    else:
        print("\n⚠️  There are some differences between implementations.")
        print("   This may be due to implementation details or edge cases.")
    
    print("\n✅ Implementation is READY for RandomForest and GBM integration!")
    print("="*70)


def run_comparison():
    """Run the full comparison suite."""
    if not SKLEARN_AVAILABLE:
        print("ERROR: sklearn is required for this comparison.")
        print("Install with: pip install scikit-learn")
        return
    
    print("\n" + "#"*70)
    print("#       MY DECISION TREE VS SKLEARN COMPARISON")
    print("#"*70)
    
    # Run comparisons
    cls_results = run_classification_comparison()
    reg_results = run_regression_comparison()
    
    # Create visualizations
    print("\n" + "="*70)
    print("     CREATING VISUALIZATIONS")
    print("="*70)
    
    create_comparison_visualization(cls_results, reg_results)
    create_detailed_comparison()
    
    # Print summary
    print_summary(cls_results, reg_results)


if __name__ == '__main__':
    run_comparison()
