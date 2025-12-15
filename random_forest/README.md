# 🌲 Random Forest Implementation Guide

**Owner: Abdellah (@AbdellahBaqua)**

## Your Task
Implement **Random Forest** based on **Breiman (2001)**: *"Random Forests"*

## 📋 Requirements

### Class Signature
```python
class RandomForest:
    def __init__(
        self,
        n_estimators=100,         # Number of trees
        max_depth=None,           # Max depth per tree
        min_samples_split=2,      # Min samples to split
        min_samples_leaf=1,       # Min samples in leaf
        max_features='sqrt',      # Features per split (MUST be 'sqrt' for RF)
        bootstrap=True,           # Bootstrap sampling (MUST be True for RF)
        oob_score=False,          # Out-of-bag score
        random_state=None,        # Reproducibility
        n_jobs=None               # Parallel jobs (optional)
    ):
        pass
    
    def fit(self, X, y):
        """Train the random forest."""
        pass
    
    def predict(self, X):
        """Predict class labels (majority vote) or values (average)."""
        pass
    
    def predict_proba(self, X):
        """Predict class probabilities (average across trees)."""
        pass
```

## 🔑 Key Breiman (2001) Requirements

### 1. Bootstrap Sampling
```python
# For each tree, sample n points WITH replacement
n_samples = X.shape[0]
bootstrap_indices = rng.choice(n_samples, size=n_samples, replace=True)
X_bootstrap = X[bootstrap_indices]
y_bootstrap = y[bootstrap_indices]
```

### 2. Random Feature Selection
```python
# Use max_features='sqrt' - the decision tree handles this!
tree = DecisionTree(
    max_depth=self.max_depth,
    max_features='sqrt',       # <-- This enables random feature selection
    random_state=tree_seed
)
```

### 3. No Pruning
> "In random forests, each tree is grown to the largest extent possible" - Breiman 2001

Set `max_depth=None` for fully grown trees (unless user specifies otherwise).

### 4. Aggregation
- **Classification**: Majority vote across all trees
- **Regression**: Average predictions across all trees

## 📁 File Structure
```
random_forest/
├── __init__.py
├── forest.py           # Main RandomForest class
└── tests.py            # Your tests
```

## 💡 Using the Decision Tree

```python
import sys
sys.path.insert(0, '..')
from decision_tree import DecisionTree

# The DecisionTree is already configured for RF mode!
tree = DecisionTree(
    max_depth=self.max_depth,
    min_samples_split=self.min_samples_split,
    min_samples_leaf=self.min_samples_leaf,
    max_features='sqrt',           # RANDOMIZED mode for RF
    criterion='gini',              # or 'entropy' for classification
    random_state=seed
)
tree.fit(X_bootstrap, y_bootstrap)
```

## ✅ Implementation Checklist

- [ ] Bootstrap sampling for each tree
- [ ] Use `max_features='sqrt'` (handled by DecisionTree)
- [ ] Store all trees in a list
- [ ] Implement majority vote (classification)
- [ ] Implement averaging (regression)
- [ ] Auto-detect task type from y
- [ ] OOB score calculation (optional but recommended)
- [ ] `feature_importances_` attribute (average across trees)
- [ ] Write unit tests

## 🧪 Test Your Implementation

```python
from sklearn.datasets import make_classification
from random_forest import RandomForest

X, y = make_classification(n_samples=500, n_features=20, random_state=42)
rf = RandomForest(n_estimators=100, random_state=42)
rf.fit(X, y)
accuracy = (rf.predict(X) == y).mean()
print(f"Training accuracy: {accuracy:.4f}")
```

## 📚 References
- Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
- Section 3: "Random Features" - explains `max_features='sqrt'`
- Section 4: "Random Forests" - full algorithm

## ⚠️ Important Notes

1. **DO NOT** modify the `decision_tree/` module (owned by Wiam)
2. **DO NOT** modify the `gradient_boosting/` module (owned by Saif)
3. **DO** use `max_features='sqrt'` for proper random forests
4. **DO** use bootstrap sampling (`replace=True`)
5. Create your implementation in `random_forest/` folder
6. Run CI tests before pushing

Good luck Abdellah! 🍀
