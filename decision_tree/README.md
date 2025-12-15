# 🌳 Decision Tree - CART Implementation

A complete **CART-style Decision Tree** implementation in Python for both **classification** and **regression** tasks. This implementation follows Breiman (2001) specifications and is designed to be used as a base learner for **Random Forest** and **Gradient Boosting Machine (GBM)** algorithms.

---

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [Usage Examples](#-usage-examples)
- [Module Architecture](#-module-architecture)
- [Parameters Guide](#-parameters-guide)
- [For Random Forest / GBM Integration](#-for-random-forest--gbm-integration)
- [Testing](#-testing)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Classification** | Gini impurity, Entropy criteria |
| **Regression** | Mean Squared Error (MSE) criterion |
| **Two Modes** | DETERMINISTIC (GBM) and RANDOMIZED (RF) |
| **Auto Task Detection** | Automatically detects classification vs regression |
| **Feature Importance** | Built-in feature importance calculation |
| **Vectorized Operations** | Fast NumPy-based computations |
| **Memory Efficient** | Uses `__slots__` for optimized memory usage |
| **Sklearn Compatible** | Similar API to scikit-learn DecisionTreeClassifier |

---

## 📦 Installation

No external installation required! Just ensure you have NumPy:

```bash
pip install numpy
```

---

## 🚀 Quick Start

### Classification

```python
from decision_tree import DecisionTree
import numpy as np

# Sample data
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]])
y = np.array([0, 0, 1, 1, 0, 1])

# Create and train
tree = DecisionTree(max_depth=3, criterion='gini')
tree.fit(X, y)

# Predict
predictions = tree.predict(X)
print(f"Predictions: {predictions}")

# Probabilities
probas = tree.predict_proba(X)
print(f"Probabilities: {probas}")
```

### Regression

```python
from decision_tree import DecisionTree
import numpy as np

# Sample data
X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([1.0, 2.1, 2.9, 4.2, 5.1, 5.8])

# Create and train
tree = DecisionTree(criterion='mse', max_depth=3)
tree.fit(X, y)

# Predict
predictions = tree.predict(X)
print(f"Predictions: {predictions}")
```

---

## 📖 API Reference

### DecisionTree Class

```python
from decision_tree import DecisionTree

tree = DecisionTree(
    max_depth=None,          # Maximum tree depth (None = unlimited)
    min_samples_split=2,     # Min samples to split a node
    min_samples_leaf=1,      # Min samples at a leaf
    criterion='gini',        # 'gini', 'entropy', or 'mse'
    max_features=None,       # Features to consider per split
    random_state=None        # Random seed
)
```

### Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `fit(X, y)` | Train the decision tree | `self` |
| `predict(X)` | Predict class labels or values | `ndarray` |
| `predict_proba(X)` | Predict class probabilities | `ndarray` |
| `get_depth()` | Get maximum depth of tree | `int` |
| `get_n_leaves()` | Get number of leaf nodes | `int` |

### Attributes (after fitting)

| Attribute | Description |
|-----------|-------------|
| `root_` | Root node of the tree |
| `n_features_` | Number of features |
| `n_classes_` | Number of classes (classification) |
| `classes_` | Unique class labels (classification) |
| `task_type_` | 'classification' or 'regression' |
| `feature_importances_` | Feature importance scores |

---

## 💡 Usage Examples

### 1. Binary Classification with Gini Impurity

```python
from decision_tree import DecisionTree
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate data
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
tree = DecisionTree(max_depth=5, criterion='gini')
tree.fit(X_train, y_train)

# Evaluate
predictions = tree.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy:.2%}")
```

### 2. Multi-class Classification with Entropy

```python
from decision_tree import DecisionTree
from sklearn.datasets import load_iris

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Train
tree = DecisionTree(max_depth=4, criterion='entropy')
tree.fit(X, y)

# Predict probabilities
probas = tree.predict_proba(X[:5])
print(f"Class probabilities:\n{probas}")
```

### 3. Regression with MSE

```python
from decision_tree import DecisionTree
import numpy as np

# Generate sinusoidal data
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X.ravel()) + np.random.normal(0, 0.1, 100)

# Train
tree = DecisionTree(criterion='mse', max_depth=6)
tree.fit(X, y)

# Predict
predictions = tree.predict(X)
mse = np.mean((predictions - y) ** 2)
print(f"MSE: {mse:.4f}")
```

### 4. Feature Importance Analysis

```python
from decision_tree import DecisionTree
import numpy as np

# Train tree
tree = DecisionTree(max_depth=5)
tree.fit(X_train, y_train)

# Get feature importances
importances = tree.feature_importances_

# Display
for i, imp in enumerate(importances):
    print(f"Feature {i}: {imp:.4f}")
```

### 5. Random Forest Mode (Feature Subsampling)

```python
from decision_tree import DecisionTree

# Use max_features='sqrt' for Random Forest style training
tree = DecisionTree(
    max_depth=None,           # Unpruned tree (Breiman default)
    criterion='gini',
    max_features='sqrt',      # Random feature subset
    random_state=42
)
tree.fit(X_train, y_train)
```

### 6. GBM Mode (Deterministic)

```python
from decision_tree import DecisionTree

# Use max_features=None for GBM style training
tree = DecisionTree(
    max_depth=3,              # Shallow trees for GBM
    criterion='mse',          # MSE for gradient boosting
    max_features=None,        # Consider ALL features
    min_samples_leaf=1
)
tree.fit(X_train, residuals)
```

---

## 🏗️ Module Architecture

```
decision_tree/
├── __init__.py      # Package exports
├── tree.py          # Main DecisionTree class
├── node.py          # Node class (5 attributes)
├── splitter.py      # Vectorized split finding
├── metrics.py       # Impurity functions (gini, entropy, mse)
├── tests.py         # Unit tests
└── README.md        # This file
```

### Module Descriptions

| Module | Purpose |
|--------|---------|
| **tree.py** | Main `DecisionTree` class with fit/predict methods |
| **node.py** | `Node` class with exactly 5 attributes for RF/GBM compatibility |
| **splitter.py** | `Splitter` class for finding optimal splits (vectorized) |
| **metrics.py** | Impurity functions: `gini()`, `entropy()`, `mse()` |

---

## ⚙️ Parameters Guide

### `max_depth`
- `None` (default): Grow tree until all leaves are pure or min_samples constraints
- `int`: Maximum depth limit

```python
# Unpruned tree (Random Forest style)
tree = DecisionTree(max_depth=None)

# Shallow tree (GBM style)
tree = DecisionTree(max_depth=3)
```

### `criterion`
| Value | Task | Formula |
|-------|------|---------|
| `'gini'` | Classification | $1 - \sum p_i^2$ |
| `'entropy'` | Classification | $-\sum p_i \log_2(p_i)$ |
| `'mse'` | Regression | $\frac{1}{n}\sum(y_i - \bar{y})^2$ |

### `max_features`
| Value | Behavior | Use Case |
|-------|----------|----------|
| `None` | All features | GBM (deterministic) |
| `'sqrt'` | $\sqrt{n_{features}}$ | Random Forest |
| `int` | Exact number | Custom |
| `float` | Fraction (0-1) | Custom |

```python
# GBM mode - all features
tree = DecisionTree(max_features=None)

# RF mode - sqrt features
tree = DecisionTree(max_features='sqrt')

# Custom - 5 features
tree = DecisionTree(max_features=5)

# Custom - 50% of features
tree = DecisionTree(max_features=0.5)
```

---

## 🔌 For Random Forest / GBM Integration

This implementation is designed as a **black-box base learner** for ensemble methods.

### Node Structure (5 Attributes)

```python
class Node:
    feature_index: int      # -1 for leaf nodes
    threshold: float        # Split value
    left_child: Node        # Left subtree
    right_child: Node       # Right subtree
    leaf_value: float       # Prediction value
```

### Accessing Tree Structure

```python
from decision_tree import DecisionTree

tree = DecisionTree()
tree.fit(X, y)

# Access root node
root = tree.root_

# Traverse tree
def traverse(node, depth=0):
    if node.is_leaf():
        print(f"{'  '*depth}Leaf: {node.leaf_value}")
    else:
        print(f"{'  '*depth}Split: X[{node.feature_index}] <= {node.threshold}")
        traverse(node.left_child, depth + 1)
        traverse(node.right_child, depth + 1)

traverse(root)
```

### Random Forest Integration Example

```python
import numpy as np
from decision_tree import DecisionTree

class SimpleRandomForest:
    def __init__(self, n_estimators=10, max_features='sqrt', random_state=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
    
    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]
        
        for i in range(self.n_estimators):
            # Bootstrap sample
            indices = rng.choice(n_samples, n_samples, replace=True)
            X_boot, y_boot = X[indices], y[indices]
            
            # Train tree with random feature selection
            tree = DecisionTree(
                max_features=self.max_features,
                random_state=rng.randint(0, 10000)
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
        
        return self
    
    def predict(self, X):
        # Majority voting
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(
            lambda x: np.bincount(x.astype(int)).argmax(), 
            axis=0, 
            arr=predictions
        )
```

### GBM Integration Example

```python
import numpy as np
from decision_tree import DecisionTree

class SimpleGBM:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.init_prediction = None
    
    def fit(self, X, y):
        # Initial prediction (mean)
        self.init_prediction = np.mean(y)
        predictions = np.full(len(y), self.init_prediction)
        
        for _ in range(self.n_estimators):
            # Compute residuals
            residuals = y - predictions
            
            # Fit tree to residuals (deterministic mode)
            tree = DecisionTree(
                max_depth=self.max_depth,
                criterion='mse',
                max_features=None  # GBM uses ALL features
            )
            tree.fit(X, residuals)
            self.trees.append(tree)
            
            # Update predictions
            predictions += self.learning_rate * tree.predict(X)
        
        return self
    
    def predict(self, X):
        predictions = np.full(X.shape[0], self.init_prediction)
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return predictions
```

---

## 🧪 Testing

Run the built-in tests:

```bash
cd decision_tree
python tests.py
```

Or from the project root:

```bash
python tests/test_classification_tree.py
python tests/test_regression_tree.py
python tests/my_tree_vs_sklearn.py
```

---

## 📊 Performance Comparison with Sklearn

```python
from decision_tree import DecisionTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
import time

# Generate data
X, y = make_classification(n_samples=10000, n_features=20, random_state=42)

# Our implementation
start = time.time()
our_tree = DecisionTree(max_depth=10)
our_tree.fit(X, y)
our_time = time.time() - start
our_acc = np.mean(our_tree.predict(X) == y)

# Sklearn
start = time.time()
sk_tree = DecisionTreeClassifier(max_depth=10)
sk_tree.fit(X, y)
sk_time = time.time() - start
sk_acc = np.mean(sk_tree.predict(X) == y)

print(f"Our Tree   - Accuracy: {our_acc:.2%}, Time: {our_time:.3f}s")
print(f"Sklearn    - Accuracy: {sk_acc:.2%}, Time: {sk_time:.3f}s")
```

---

## 📚 References

- Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32.
- Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984). *Classification and Regression Trees*. CRC press.

---

## 👥 Authors

**Wiam** - Decision Tree Implementation (Member 1)

Part of the **RandomForest-GBM** project.

---

## 📄 License

MIT License - See [LICENSE](../LICENSE) for details.
