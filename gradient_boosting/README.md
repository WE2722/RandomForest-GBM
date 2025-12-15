# Gradient Boosting Machine (GBM)

**A Complete Implementation of Gradient Boosting for Regression and Classification**

*Author: Saif (@Saif-dbot)*

---

## Table of Contents

1. [Introduction](#introduction)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Implementation](#implementation)
4. [Usage](#usage)
5. [Performance & Benchmarks](#performance--benchmarks)
6. [References](#references)

---

## Introduction

This repository provides a complete, from-scratch implementation of **Gradient Boosting Machines (GBM)**, a powerful ensemble learning technique that builds models sequentially by optimizing a loss function through gradient descent in function space.

### Key Features

The implementation includes a complete regressor and classifier supporting multiple loss functions. The API is compatible with scikit-learn for seamless integration. A comprehensive test suite validates the implementation against scikit-learn benchmarks. Extensive mathematical documentation is provided with full derivations. The system handles binary and multiclass classification seamlessly.

### Implemented Components

| Component | Loss Functions | Status |
|-----------|---------------|--------|
| GradientBoostingRegressor | squared_error, absolute_error | Implemented |
| GradientBoostingClassifier | log_loss (binary and multiclass) | Implemented |
| Test Suite | Comprehensive validation | Complete |
| Documentation | Full mathematical exposition | Complete | 


---

## Mathematical Foundation

### Overview

Gradient Boosting builds an ensemble model $F_M(x)$ as a sum of weak learners, typically shallow decision trees:

$$F_M(x) = F_0(x) + \sum_{m=1}^{M} \eta \cdot h_m(x)$$

where:
- $F_0(x)$ is the initial prediction
- $h_m(x)$ are weak learners (shallow trees)
- $\eta$ is the learning rate (shrinkage parameter)
- $M$ is the number of boosting iterations

### Algorithm Principles

#### 1. **Initialization**

The initial model $F_0$ minimizes the loss over the training data:

$$F_0 = \arg\min_{\gamma} \sum_{i=1}^{n} L(y_i, \gamma)$$

For common loss functions:
- **Squared Error**: $F_0 = \frac{1}{n}\sum_{i=1}^{n} y_i$ (mean)
- **Absolute Error**: $F_0 = \text{median}(y)$
- **Log Loss (Binary)**: $F_0 = \log\left(\frac{p}{1-p}\right)$ where $p = \frac{\sum y_i}{n}$

#### 2. **Iterative Optimization**

At each iteration $m$, we perform gradient descent in function space:

**Step 1:** Compute pseudo-residuals (negative gradient):

$$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F = F_{m-1}}$$

**Step 2:** Fit a weak learner $h_m$ to the residuals:

$$h_m = \arg\min_h \sum_{i=1}^{n} (r_{im} - h(x_i))^2$$

**Step 3:** Update the ensemble:

$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

### Loss Functions and Gradients

#### Regression

**Squared Error (L2)**:
$$L(y, F) = \frac{1}{2}(y - F)^2 \quad \Rightarrow \quad r = y - F$$

**Absolute Error (L1)**:
$$L(y, F) = |y - F| \quad \Rightarrow \quad r = \text{sign}(y - F)$$

#### Binary Classification

**Log Loss (Deviance)**:
$$L(y, F) = -[y \log(p) + (1-y) \log(1-p)]$$

where $p = \frac{1}{1 + e^{-F}}$ (sigmoid)

$$r = y - p$$

#### Multiclass Classification

For $K$ classes, using **One-vs-All** with softmax:

$$p_k = \frac{e^{F_k}}{\sum_{j=1}^{K} e^{F_j}}$$

$$r_{ik} = y_{ik} - p_k(x_i)$$

where $y_{ik} = 1$ if sample $i$ belongs to class $k$, else 0.

---

## Implementation

### Architecture

```
gradient_boosting/
├── gbm.py                   # Base GBM class
├── regressor.py             # GradientBoostingRegressor
├── classifier.py            # GradientBoostingClassifier
├── test_gbm.py              # Test suite
└── README.md                # This file
```

### API Reference

#### Class: `GradientBoostingRegressor` / `GradientBoostingClassifier`

```python
class GradientBoostingRegressor:
    """
    Gradient Boosting for regression tasks.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting stages (weak learners)
        
    learning_rate : float, default=0.1
        Shrinkage parameter η ∈ (0, 1]. Controls overfitting.
        Smaller values require more estimators.
        
    max_depth : int, default=3
        Maximum depth of individual trees. Typically shallow (3-8).
        
    min_samples_split : int, default=2
        Minimum samples required to split a node
        
    min_samples_leaf : int, default=1
        Minimum samples required in a leaf node
        
    subsample : float, default=1.0
        Fraction of samples used per tree (stochastic GB).
        Values < 1.0 enable stochastic gradient boosting.
        
    loss : {'squared_error', 'absolute_error'}, default='squared_error'
        Loss function to optimize
        
    random_state : int, optional
        Random seed for reproducibility
    """
    
    def fit(self, X, y):
        """
        Train the gradient boosting model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : object
            Fitted estimator
        """
    
    def predict(self, X):
        """
        Predict using the gradient boosting model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values
        """
    
    def staged_predict(self, X):
        """
        Yield predictions at each boosting iteration.
        
        Useful for determining optimal number of estimators
        and creating learning curves.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples
            
        Yields
        ------
        y_pred : ndarray of shape (n_samples,)
            Predictions after each boosting iteration
        """
```

#### Classifier-Specific Methods

```python
class GradientBoostingClassifier:
    """
    Additional methods for classification
    """
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples
            
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities
        """
    
    def staged_predict_proba(self, X):
        """
        Yield class probabilities at each boosting iteration.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples
            
        Yields
        ------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities after each iteration
        """
```


#### Pseudocode

```
Algorithm: Gradient Boosting Machine

Input: Training data (X, y), loss function L, number of iterations M
Output: Ensemble model F_M

1. Initialize F_0:
   F_0 ← argmin_γ Σ L(y_i, γ)

2. For m = 1 to M:
   a. Compute pseudo-residuals:
      r_im ← -∂L(y_i, F(x_i))/∂F(x_i)|_{F=F_{m-1}}
   
   b. Fit regression tree h_m to residuals:
      h_m ← argmin_h Σ(r_im - h(x_i))²
   
   c. Update model:
      F_m(x) ← F_{m-1}(x) + η·h_m(x)

3. Return F_M(x)
```

---

## Usage

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd RandomForest-GBM

# The module uses relative imports
# Add parent directory to Python path or install as package
```

### Quick Start
├── classifier.py            # GradientBoostingClassifier
├── test_gbm.py              # Comprehensive test suite
├── README.md                # This file
└── GBM_EXPLANATION.pdf      # Full documentation
```


#### Regression Example

```python
import numpy as np
from gradient_boosting import GradientBoostingRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X, y = make_regression(n_samples=500, n_features=10, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"Feature Importances: {model.feature_importances_[:5]}")
```

### Binary Classification Example

```python
from gradient_boosting import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

gbc = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gbc.fit(X_train, y_train)

y_pred = gbc.predict(X_test)
y_proba = gbc.predict_proba(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
```

### Multiclass Classification Example

```python
from sklearn.datasets import load_iris

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

gbc = GradientBoostingClassifier(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gbc.fit(X_train, y_train)

y_pred = gbc.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Classes: {gbc.classes_}")
```

### Learning Curves with Staged Predictions

```python
import matplotlib.pyplot as plt

gbm = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gbm.fit(X_train, y_train)

train_errors = []
test_errors = []

for train_pred, test_pred in zip(gbm.staged_predict(X_train), gbm.staged_predict(X_test)):
    train_errors.append(mean_squared_error(y_train, train_pred))
    test_errors.append(mean_squared_error(y_test, test_pred))

plt.figure(figsize=(10, 6))
plt.plot(train_errors, label='Training Error')
plt.plot(test_errors, label='Test Error')
plt.xlabel('Boosting Iterations')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('GBM Learning Curves')
plt.show()
```


### Loss Functions

#### Regression
```python
loss = 0.5 * (y - F(x))^2
negative_gradient = y - F(x)

loss = |y - F(x)|
negative_gradient = sign(y - F(x))
```

#### Classification (Binary)
```python
loss = -[y*log(p) + (1-y)*log(1-p)]
negative_gradient = y - p
```


## Test Results Summary

All tests passing with excellent performance:
- Regression: MSE < 1.0 on synthetic data
- Binary Classification: Accuracy > 95%
- Multiclass Classification: Accuracy > 99%
- sklearn Comparison: <10% difference in metrics
- Feature Importances: Properly normalized (sum = 1.0)
- Staged Predictions: Consistent with final predictions

## References
- Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine.
- Friedman, J. H. (2002). Stochastic Gradient Boosting.
- Section 4 of Friedman (2001): Algorithm details

## Package Integration Notes

### Dependencies
- `numpy`: Core numerical operations
- `decision_tree`: Base tree implementation (from parent module)
- `sklearn` (optional): For testing and comparison only

### Module Structure for Package
```python
from .regressor import GradientBoostingRegressor
from .classifier import GradientBoostingClassifier
from .gbm import GradientBoostingMachine

__all__ = [
    'GradientBoostingRegressor',
    'GradientBoostingClassifier',
    'GradientBoostingMachine'
]
```

### Important Implementation Details

1. **Deterministic Splits**: Uses `max_features=None` for reproducible and optimal splits (unlike Random Forest)
2. **Shallow Trees**: Default `max_depth=3` provides best bias-variance tradeoff
3. **Sequential Training**: Trees are trained sequentially, not in parallel
4. **Feature Importances**: Accumulated across all trees and normalized to sum to 1.0
5. **Loss Functions**:
   - Regression: squared_error (L2), absolute_error (L1)
   - Classification: log_loss (binary and multiclass with softmax)
6. **Numerical Stability**: All probability calculations use clipping to avoid log(0)

### Integration with decision_tree Module

```python
from decision_tree import DecisionTree
```

### API Compatibility

The implementation follows sklearn conventions:
- `.fit(X, y)` returns self for method chaining
- `.predict(X)` returns predictions
- `.predict_proba(X)` returns class probabilities (classifier only)
- `.staged_predict(X)` yields predictions at each stage
- `._` suffix for fitted attributes (estimators_, init_, etc.)

