"""
Random Forest Comparison Tests.

This module compares Random Forest with Decision Trees.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from random_forest import RandomForest


def test_effect_of_n_estimators():
    """Test effect of number of trees on performance."""
    print("\n" + "="*60)
    print("TEST 1: Effect of Number of Trees (n_estimators)")
    print("="*60)
    
    # Generate classification data
    np.random.seed(42)
    n_samples = 200
    
    X0 = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])
    X1 = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
    
    X = np.vstack([X0, X1])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    
    idx = np.random.permutation(n_samples)
    X, y = X[idx], y[idx]
    
    # Test different numbers of trees
    n_trees_range = [1, 3, 5, 10, 20, 50, 100]
    accuracies = []
    
    for n_trees in n_trees_range:
        rf = RandomForest(
            n_estimators=n_trees,
            max_depth=5,
            max_features='sqrt',
            random_state=42
        )
        rf.fit(X, y)
        acc = np.mean(rf.predict(X) == y)
        accuracies.append(acc)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(n_trees_range, accuracies, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Trees (n_estimators)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Effect of Number of Trees on Accuracy')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'images', 
                'rf_n_estimators_effect.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ Plot saved: rf_n_estimators_effect.png")
    
    print("\nAccuracy by number of trees:")
    for n_trees, acc in zip(n_trees_range, accuracies):
        print(f"  {n_trees:3d} trees: {acc*100:.2f}%")
    
    return accuracies


def test_variance_reduction():
    """Demonstrate variance reduction with Random Forest."""
    print("\n" + "="*60)
    print("TEST 2: Variance Reduction with Ensemble")
    print("="*60)
    
    # Generate data
    np.random.seed(42)
    n_samples = 150
    
    X0 = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])
    X1 = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
    
    X = np.vstack([X0, X1])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    
    idx = np.random.permutation(n_samples)
    X, y = X[idx], y[idx]
    
    # Train single trees with different random states
    single_tree_accs = []
    
    print("\nTraining 10 single decision trees with different random seeds...")
    from decision_tree import DecisionTree
    for i in range(10):
        tree = DecisionTree(max_depth=5, criterion='gini', random_state=i)
        tree.fit(X, y)
        acc = np.mean(tree.predict(X) == y)
        single_tree_accs.append(acc)
    
    print("Training Random Forest with 10 trees...")
    rf = RandomForest(
        n_estimators=10,
        max_depth=5,
        max_features='sqrt',
        random_state=42
    )
    rf.fit(X, y)
    rf_acc = np.mean(rf.predict(X) == y)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Single trees
    ax = axes[0]
    ax.bar(range(1, 11), single_tree_accs, color='skyblue', edgecolor='black')
    ax.axhline(y=np.mean(single_tree_accs), color='red', linestyle='--', 
              linewidth=2, label=f'Mean: {np.mean(single_tree_accs)*100:.1f}%')
    ax.set_xlabel('Tree Index')
    ax.set_ylabel('Accuracy')
    ax.set_title('Single Trees (Different Random Seeds)')
    ax.set_ylim([0.7, 1])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Random Forest
    ax = axes[1]
    ax.bar(['Random\nForest'], [rf_acc], color='lightgreen', edgecolor='black', width=0.5)
    ax.axhline(y=np.mean(single_tree_accs), color='red', linestyle='--',
              linewidth=2, label=f'Mean of single trees: {np.mean(single_tree_accs)*100:.1f}%')
    ax.set_ylabel('Accuracy')
    ax.set_title('Random Forest Ensemble')
    ax.set_ylim([0.7, 1])
    ax.text(0, rf_acc + 0.02, f'{rf_acc*100:.1f}%', ha='center', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Variance Reduction: Single Trees vs Ensemble', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'images', 
                'rf_variance_reduction.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ Plot saved: rf_variance_reduction.png")
    
    print(f"\nVariance Reduction Analysis:")
    print(f"  Single tree accuracies: {[f'{acc*100:.1f}%' for acc in single_tree_accs]}")
    print(f"  Mean accuracy: {np.mean(single_tree_accs)*100:.2f}%")
    print(f"  Std deviation: {np.std(single_tree_accs)*100:.2f}%")
    print(f"\n  Random Forest accuracy: {rf_acc*100:.2f}%")


if __name__ == '__main__':
    # Ensure images directory exists
    img_dir = os.path.join(os.path.dirname(__file__), 'images')
    os.makedirs(img_dir, exist_ok=True)
    
    test_effect_of_n_estimators()
    test_variance_reduction()
    
    print("\n" + "="*60)
    print("All Random Forest Comparison Tests Completed!")
    print("="*60)
