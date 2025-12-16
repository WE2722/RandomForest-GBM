"""
Compare Custom Random Forest vs sklearn Random Forest.

This module provides a comprehensive comparison between the custom
Random Forest implementation and sklearn's RandomForest.

Author: Member 1
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forest import RandomForest

# Import sklearn
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.datasets import make_classification, make_regression, load_iris, load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("WARNING: sklearn not available. Install with: pip install scikit-learn")


def compare_classification(X_train, X_test, y_train, y_test, dataset_name, n_estimators=50, max_depth=8):
    """
    Compare classification performance between custom forest and sklearn.
    
    Returns
    -------
    dict : Comparison results
    """
    results = {'dataset': dataset_name, 'task': 'classification'}
    
    # Custom forest
    print(f"\n{'-'*60}")
    print(f"CLASSIFICATION: {dataset_name}")
    print(f"{'-'*60}")
    print("\nCustom Random Forest:")
    
    start = time.time()
    my_forest = RandomForest(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    my_forest.fit(X_train, y_train)
    my_train_time = time.time() - start
    
    start = time.time()
    my_train_pred = my_forest.predict(X_train)
    my_test_pred = my_forest.predict(X_test)
    my_pred_time = time.time() - start
    
    my_train_acc = accuracy_score(y_train, my_train_pred)
    my_test_acc = accuracy_score(y_test, my_test_pred)
    
    print(f"  Training time:  {my_train_time:.4f}s")
    print(f"  Prediction time: {my_pred_time:.4f}s")
    print(f"  Train Accuracy: {my_train_acc:.6f}")
    print(f"  Test Accuracy:  {my_test_acc:.6f}")
    
    results['custom_train_time'] = my_train_time
    results['custom_pred_time'] = my_pred_time
    results['custom_train_acc'] = my_train_acc
    results['custom_test_acc'] = my_test_acc
    
    # sklearn forest
    print("\nsklearn Random Forest:")
    
    start = time.time()
    sk_forest = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        random_state=42,
        n_jobs=1
    )
    sk_forest.fit(X_train, y_train)
    sk_train_time = time.time() - start
    
    start = time.time()
    sk_train_pred = sk_forest.predict(X_train)
    sk_test_pred = sk_forest.predict(X_test)
    sk_pred_time = time.time() - start
    
    sk_train_acc = accuracy_score(y_train, sk_train_pred)
    sk_test_acc = accuracy_score(y_test, sk_test_pred)
    
    print(f"  Training time:  {sk_train_time:.4f}s")
    print(f"  Prediction time: {sk_pred_time:.4f}s")
    print(f"  Train Accuracy: {sk_train_acc:.6f}")
    print(f"  Test Accuracy:  {sk_test_acc:.6f}")
    
    results['sklearn_train_time'] = sk_train_time
    results['sklearn_pred_time'] = sk_pred_time
    results['sklearn_train_acc'] = sk_train_acc
    results['sklearn_test_acc'] = sk_test_acc
    
    # Comparison
    print("\nComparison:")
    print(f"  Train Accuracy Difference: {abs(my_train_acc - sk_train_acc):.6f}")
    print(f"  Test Accuracy Difference:  {abs(my_test_acc - sk_test_acc):.6f}")
    print(f"  Training Time Ratio (Custom/sklearn): {my_train_time/sk_train_time:.2f}x")
    
    results['acc_diff'] = abs(my_test_acc - sk_test_acc)
    results['speed_ratio'] = my_train_time / sk_train_time if sk_train_time > 0 else 0
    
    return results


def compare_regression(X_train, X_test, y_train, y_test, dataset_name, n_estimators=50, max_depth=8):
    """
    Compare regression performance between custom forest and sklearn.
    
    Returns
    -------
    dict : Comparison results
    """
    results = {'dataset': dataset_name, 'task': 'regression'}
    
    # Custom forest
    print(f"\n{'-'*60}")
    print(f"REGRESSION: {dataset_name}")
    print(f"{'-'*60}")
    print("\nCustom Random Forest:")
    
    start = time.time()
    my_forest = RandomForest(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    my_forest.fit(X_train, y_train)
    my_train_time = time.time() - start
    
    start = time.time()
    my_train_pred = my_forest.predict(X_train)
    my_test_pred = my_forest.predict(X_test)
    my_pred_time = time.time() - start
    
    my_train_mse = mean_squared_error(y_train, my_train_pred)
    my_test_mse = mean_squared_error(y_test, my_test_pred)
    my_train_r2 = r2_score(y_train, my_train_pred)
    my_test_r2 = r2_score(y_test, my_test_pred)
    
    print(f"  Training time:  {my_train_time:.4f}s")
    print(f"  Prediction time: {my_pred_time:.4f}s")
    print(f"  Train MSE: {my_train_mse:.6f}, R²: {my_train_r2:.6f}")
    print(f"  Test MSE:  {my_test_mse:.6f}, R²: {my_test_r2:.6f}")
    
    results['custom_train_time'] = my_train_time
    results['custom_pred_time'] = my_pred_time
    results['custom_train_mse'] = my_train_mse
    results['custom_test_mse'] = my_test_mse
    results['custom_train_r2'] = my_train_r2
    results['custom_test_r2'] = my_test_r2
    
    # sklearn forest
    print("\nsklearn Random Forest:")
    
    start = time.time()
    sk_forest = RandomForestRegressor(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        random_state=42,
        n_jobs=1
    )
    sk_forest.fit(X_train, y_train)
    sk_train_time = time.time() - start
    
    start = time.time()
    sk_train_pred = sk_forest.predict(X_train)
    sk_test_pred = sk_forest.predict(X_test)
    sk_pred_time = time.time() - start
    
    sk_train_mse = mean_squared_error(y_train, sk_train_pred)
    sk_test_mse = mean_squared_error(y_test, sk_test_pred)
    sk_train_r2 = r2_score(y_train, sk_train_pred)
    sk_test_r2 = r2_score(y_test, sk_test_pred)
    
    print(f"  Training time:  {sk_train_time:.4f}s")
    print(f"  Prediction time: {sk_pred_time:.4f}s")
    print(f"  Train MSE: {sk_train_mse:.6f}, R²: {sk_train_r2:.6f}")
    print(f"  Test MSE:  {sk_test_mse:.6f}, R²: {sk_test_r2:.6f}")
    
    results['sklearn_train_time'] = sk_train_time
    results['sklearn_pred_time'] = sk_pred_time
    results['sklearn_train_mse'] = sk_train_mse
    results['sklearn_test_mse'] = sk_test_mse
    results['sklearn_train_r2'] = sk_train_r2
    results['sklearn_test_r2'] = sk_test_r2
    
    # Comparison
    print("\nComparison:")
    print(f"  Test MSE Difference: {abs(my_test_mse - sk_test_mse):.6f}")
    print(f"  Test R² Difference:  {abs(my_test_r2 - sk_test_r2):.6f}")
    print(f"  Training Time Ratio (Custom/sklearn): {my_train_time/sk_train_time:.2f}x")
    
    results['mse_diff'] = abs(my_test_mse - sk_test_mse)
    results['r2_diff'] = abs(my_test_r2 - sk_test_r2)
    results['speed_ratio'] = my_train_time / sk_train_time if sk_train_time > 0 else 0
    
    return results


def plot_comparison_results(all_results):
    """Plot comparison results across datasets."""
    
    # Separate classification and regression results
    clf_results = [r for r in all_results if r['task'] == 'classification']
    reg_results = [r for r in all_results if r['task'] == 'regression']
    
    # Classification comparison
    if clf_results:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        datasets = [r['dataset'] for r in clf_results]
        custom_accs = [r['custom_test_acc'] for r in clf_results]
        sklearn_accs = [r['sklearn_test_acc'] for r in clf_results]
        
        x = np.arange(len(datasets))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, custom_accs, width, label='Custom', alpha=0.8)
        axes[0, 0].bar(x + width/2, sklearn_accs, width, label='sklearn', alpha=0.8)
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Classification - Test Accuracy')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(datasets, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Training time comparison
        custom_times = [r['custom_train_time'] for r in clf_results]
        sklearn_times = [r['sklearn_train_time'] for r in clf_results]
        
        axes[0, 1].bar(x - width/2, custom_times, width, label='Custom', alpha=0.8)
        axes[0, 1].bar(x + width/2, sklearn_times, width, label='sklearn', alpha=0.8)
        axes[0, 1].set_ylabel('Time (seconds)')
        axes[0, 1].set_title('Classification - Training Time')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(datasets, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Accuracy differences
        acc_diffs = [r['acc_diff'] for r in clf_results]
        axes[1, 0].bar(datasets, acc_diffs, alpha=0.8, color='orange')
        axes[1, 0].set_ylabel('Absolute Difference')
        axes[1, 0].set_title('Classification - Accuracy Difference (|Custom - sklearn|)')
        axes[1, 0].set_xticklabels(datasets, rotation=45)
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Speed ratio
        speed_ratios = [r['speed_ratio'] for r in clf_results]
        colors = ['green' if r < 1 else 'red' for r in speed_ratios]
        axes[1, 1].bar(datasets, speed_ratios, alpha=0.8, color=colors)
        axes[1, 1].axhline(y=1, color='black', linestyle='--', label='Equal Speed')
        axes[1, 1].set_ylabel('Time Ratio')
        axes[1, 1].set_title('Classification - Training Time Ratio (Custom/sklearn)')
        axes[1, 1].set_xticklabels(datasets, rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('comparison_classification.png', dpi=150)
        print("\n✓ Plot saved: comparison_classification.png")
        plt.close()
    
    # Regression comparison
    if reg_results:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        datasets = [r['dataset'] for r in reg_results]
        custom_r2s = [r['custom_test_r2'] for r in reg_results]
        sklearn_r2s = [r['sklearn_test_r2'] for r in reg_results]
        
        x = np.arange(len(datasets))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, custom_r2s, width, label='Custom', alpha=0.8)
        axes[0, 0].bar(x + width/2, sklearn_r2s, width, label='sklearn', alpha=0.8)
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_title('Regression - Test R² Score')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(datasets, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Training time comparison
        custom_times = [r['custom_train_time'] for r in reg_results]
        sklearn_times = [r['sklearn_train_time'] for r in reg_results]
        
        axes[0, 1].bar(x - width/2, custom_times, width, label='Custom', alpha=0.8)
        axes[0, 1].bar(x + width/2, sklearn_times, width, label='sklearn', alpha=0.8)
        axes[0, 1].set_ylabel('Time (seconds)')
        axes[0, 1].set_title('Regression - Training Time')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(datasets, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # MSE differences
        mse_diffs = [r['mse_diff'] for r in reg_results]
        axes[1, 0].bar(x, mse_diffs, alpha=0.8, color='purple')
        axes[1, 0].set_ylabel('Absolute Difference')
        axes[1, 0].set_title('Regression - MSE Difference (|Custom - sklearn|)')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(datasets, rotation=45)
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Speed ratio
        speed_ratios = [r['speed_ratio'] for r in reg_results]
        colors = ['green' if r < 1 else 'red' for r in speed_ratios]
        axes[1, 1].bar(x, speed_ratios, alpha=0.8, color=colors)
        axes[1, 1].axhline(y=1, color='black', linestyle='--', label='Equal Speed')
        axes[1, 1].set_ylabel('Time Ratio')
        axes[1, 1].set_title('Regression - Training Time Ratio (Custom/sklearn)')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(datasets, rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('comparison_regression.png', dpi=150)
        print("\n✓ Plot saved: comparison_regression.png")
        plt.close()


if __name__ == "__main__":
    if not SKLEARN_AVAILABLE:
        print("\nCannot run comparisons without sklearn. Install with: pip install scikit-learn")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("RANDOM FOREST COMPARISON: Custom vs sklearn")
    print("="*70)
    
    all_results = []
    
    try:
        # CLASSIFICATION TESTS
        print("\n" + "="*70)
        print("CLASSIFICATION COMPARISONS")
        print("="*70)
        
        # Iris
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.3, random_state=42
        )
        results = compare_classification(X_train, X_test, y_train, y_test, 
                                        "Iris", n_estimators=50, max_depth=8)
        all_results.append(results)
        
        # Synthetic classification
        X, y = make_classification(n_samples=500, n_features=20, n_informative=15,
                                  n_redundant=5, n_classes=3, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        results = compare_classification(X_train, X_test, y_train, y_test, 
                                        "Synthetic", n_estimators=50, max_depth=10)
        all_results.append(results)
        
        # REGRESSION TESTS
        print("\n" + "="*70)
        print("REGRESSION COMPARISONS")
        print("="*70)
        
        # Diabetes
        diabetes = load_diabetes()
        X_train, X_test, y_train, y_test = train_test_split(
            diabetes.data, diabetes.target, test_size=0.3, random_state=42
        )
        results = compare_regression(X_train, X_test, y_train, y_test, 
                                    "Diabetes", n_estimators=50, max_depth=10)
        all_results.append(results)
        
        # Synthetic regression
        X, y = make_regression(n_samples=500, n_features=20, n_informative=15,
                              noise=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        results = compare_regression(X_train, X_test, y_train, y_test, 
                                    "Synthetic", n_estimators=50, max_depth=10)
        all_results.append(results)
        
        # Generate comparison plots
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        plot_comparison_results(all_results)
        
        # Final summary
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        print(f"\n✓ Tested on {len(all_results)} datasets")
        print("✓ Generated comparison plots:")
        print("  - comparison_classification.png")
        print("  - comparison_regression.png")
        
    except Exception as e:
        print(f"\n✗ Error during comparison: {e}")
        import traceback
        traceback.print_exc()
