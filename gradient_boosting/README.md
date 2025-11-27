# 📈 Gradient Boosting Machine Implementation Guide

**Owner: Saif (@Saif-dbot)**

## Your Task
Implement **Gradient Boosting Machine (GBM)** based on:
- Friedman (2001): *"Greedy Function Approximation: A Gradient Boosting Machine"*
- Breiman (2001) for decision tree base learners

## 📋 Requirements

### Class Signature
```python
class GradientBoostingMachine:
    def __init__(
        self,
        n_estimators=100,         # Number of boosting stages
        learning_rate=0.1,        # Shrinkage parameter (η)
        max_depth=3,              # Depth of each tree (typically shallow!)
        min_samples_split=2,      # Min samples to split
        min_samples_leaf=1,       # Min samples in leaf
        subsample=1.0,            # Fraction of samples per tree (stochastic GB)
        loss='squared_error',     # Loss function
        random_state=None         # Reproducibility
    ):
        pass
    
    def fit(self, X, y):
        """Train the gradient boosting model."""
        pass
    
    def predict(self, X):
        """Predict values."""
        pass
    
    def staged_predict(self, X):
        """Yield predictions at each stage (for learning curves)."""
        pass
```

## 🔑 Key Algorithm Steps

### 1. Initialize with Constant (F₀)
```python
# For squared error loss:
F_0 = np.mean(y)  # Initial prediction is just the mean

# For classification (log loss):
# F_0 = log(p / (1-p)) where p is the positive class proportion
```

### 2. Gradient Boosting Loop
```python
for m in range(n_estimators):
    # Step 1: Compute pseudo-residuals (negative gradient)
    residuals = y - current_predictions  # For squared error
    
    # Step 2: Fit a regression tree to residuals
    tree = DecisionTree(
        max_depth=self.max_depth,      # Usually shallow (3-8)
        max_features=None,             # DETERMINISTIC mode for GBM!
        criterion='mse',               # Always MSE for residuals
        random_state=seed
    )
    tree.fit(X, residuals)
    
    # Step 3: Update predictions with shrinkage
    current_predictions += learning_rate * tree.predict(X)
    
    # Store tree
    self.estimators_.append(tree)
```

### 3. Prediction
```python
def predict(self, X):
    # Start with initial prediction
    y_pred = np.full(X.shape[0], self.init_)
    
    # Add contribution from each tree
    for tree in self.estimators_:
        y_pred += self.learning_rate * tree.predict(X)
    
    return y_pred
```

## 📁 File Structure
```
gradient_boosting/
├── __init__.py
├── gbm.py              # Main GBM class
├── losses.py           # Loss functions (optional)
└── tests.py            # Your tests
```

## 💡 Using the Decision Tree

```python
import sys
sys.path.insert(0, '..')
from decision_tree import DecisionTree

# The DecisionTree in DETERMINISTIC mode for GBM
tree = DecisionTree(
    max_depth=self.max_depth,      # Shallow trees (3-8)
    min_samples_split=self.min_samples_split,
    min_samples_leaf=self.min_samples_leaf,
    max_features=None,             # DETERMINISTIC mode - use ALL features
    criterion='mse',               # Always fit to residuals with MSE
    random_state=seed
)
tree.fit(X_subset, residuals)
```

## ⚠️ Key Differences from Random Forest

| Aspect | Random Forest | GBM |
|--------|---------------|-----|
| `max_features` | `'sqrt'` (randomized) | `None` (deterministic) |
| Tree depth | Deep (unlimited) | Shallow (3-8) |
| Training | Parallel (independent) | Sequential (dependent) |
| Aggregation | Averaging | Additive (with learning rate) |
| Sampling | Bootstrap | Subsample (optional) |

## 📊 Loss Functions

### Regression
```python
# Squared Error (L2)
loss = 0.5 * (y - F(x))^2
negative_gradient = y - F(x)  # residuals

# Absolute Error (L1)
loss = |y - F(x)|
negative_gradient = sign(y - F(x))
```

### Classification (Binary)
```python
# Log Loss (Deviance)
loss = -[y*log(p) + (1-y)*log(1-p)]
negative_gradient = y - p  # where p = sigmoid(F(x))
```

## ✅ Implementation Checklist

- [ ] Initialize F₀ (mean for regression)
- [ ] Compute pseudo-residuals at each stage
- [ ] Use `max_features=None` (deterministic mode)
- [ ] Use shallow trees (`max_depth=3` default)
- [ ] Apply learning rate (shrinkage)
- [ ] Store all trees in order
- [ ] Implement `staged_predict()` for learning curves
- [ ] Support `subsample < 1.0` for stochastic GB
- [ ] `feature_importances_` attribute
- [ ] Write unit tests

## 🧪 Test Your Implementation

```python
from sklearn.datasets import make_regression
from gradient_boosting import GradientBoostingMachine

X, y = make_regression(n_samples=500, n_features=10, noise=10, random_state=42)
gbm = GradientBoostingMachine(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbm.fit(X, y)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y, gbm.predict(X))
print(f"Training MSE: {mse:.4f}")
```

## 📚 References
- Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine.
- Friedman, J. H. (2002). Stochastic Gradient Boosting.
- Section 4 of Friedman (2001): Algorithm details

## ⚠️ Important Notes

1. **DO NOT** modify the `decision_tree/` module (owned by Wiam)
2. **DO NOT** modify the `random_forest/` module (owned by Abdellah)
3. **DO** use `max_features=None` for GBM (deterministic splits)
4. **DO** use shallow trees (max_depth=3 to 8)
5. **DO** apply shrinkage (learning_rate)
6. Create your implementation in `gradient_boosting/` folder
7. Run CI tests before pushing

Good luck Saif! 🍀
