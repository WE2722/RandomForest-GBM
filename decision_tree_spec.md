# DecisionTree Implementation Guide - MUST FOLLOW EXACTLY

## ⚠ CRITICAL: This class is the SINGLE DEPENDENCY for RandomForest and GBM
*Your teammates (Members 2 & 3) will import this EXACT class. Any deviation breaks integration.*

## 📋 EXACT SPECIFICATIONS FROM PROJECT TASK DOCUMENT

### Class Signature (DO NOT CHANGE)
class DecisionTree:
def init(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
criterion='gini', max_features=None, random_state=None):
# Parameters EXACTLY as shown above
# criterion: 'gini', 'entropy' (classification) | 'mse' (regression)
# max_features: None=all, 'sqrt', int, float(0-1 fraction)

text
def fit(self, X, y):  # Returns self
def predict(self, X): # Returns predictions
text

### Node Structure (EXACT attributes required)
class Node:
feature_index: int # -1 for leaf
threshold: float # split value
left_child: Node # or None
right_child: Node # or None
leaf_value: float # prediction (class or mean)

text

### Impurity Functions (implement these EXACTLY)
*Classification:*
- gini(p): ∑p_i*(1-p_i) [attached_file:1]
- entropy(p): -∑p_i*log2(p_i) [attached_file:1]

*Regression:*
- mse(y): mean((y - mean(y))^2) [attached_file:1]

### TWO MODES - CRITICAL FOR TEAM INTEGRATION
S_ISL_SR_S_ISR where I is chosen impurity

DETERMINISTIC (GBM mode): max_features=None or 1.0

Search ALL features at every split

RANDOMIZED (RF mode): max_features='sqrt', int, float

Sample min(max_features, n_features) WITHOUT replacement per split

text

### Best-Split Algorithm (vectorized NumPy REQUIRED)
For each node:

Check stopping: max_depth, min_samples_split, min_samples_leaf, gain<=0 → LEAF

Select features:

Deterministic: all features (0:n_features)

Randomized: np.random.choice(n_features, size=min(max_features, n_features), replace=False)

For each feature:

thresholds = np.unique(X[:,feat])[1:] OR midpoints between sorted unique

For each threshold (VECTORIZED):
left_mask = X[:,feat] <= threshold
gain = I_parent - (n_left/n * I_left + n_right/n * I_right)

Pick BEST (feature, threshold) with max gain > 0

Split & recurse

text

### Stopping Conditions (ALL must be implemented)
- max_depth reached (count from root)
- node.n_samples < min_samples_split
- Both children would have < min_samples_leaf
- best_gain <= 0 (no improvement)

### Leaf Values
- *Classification*: argmax(class_counts)
- *Regression*: np.mean(y)

## 🔧 IMPLEMENTATION REQUIREMENTS

### Auto-detect task type in fit():
if len(np.unique(y)) <= 20 and not np.issubdtype(y.dtype, np.floating):
task = 'classification' # criterion='gini'/'entropy'
else:
task = 'regression' # criterion='mse'

text

### Random State (REPRODUCIBLE REQUIRED)
if random_state is not None:
np.random.seed(random_state)

Use for feature subsampling ONLY
text

### Efficiency (MANDATORY)
- *Vectorized NumPy* - NO Python loops for split evaluation
- O(n_samples * n_features * log(n_samples)) target
- Use np.unique(X[:, feat])[1:] for thresholds
- *Optional*: midpoints (sorted_unique[:-1] + sorted_unique[1:])/2

## 🧪 REQUIRED EDGE CASES
✅ Pure node (all same class/value) → immediate leaf
✅ Single sample → leaf
✅ All identical features → no split
✅ max_features=0 → no split
✅ min_samples_split=1 → always split if gain>0
✅ Randomized reproducible with same random_state

text

## 📄 REFERENCE PAPERS - IMPLEMENT EXACTLY AS DESCRIBED

### Random Forest (Breiman 2001) [attached_file:2]
Feature subsampling at each node (Section 4)

max_features = sqrt(M) default for classification

Unpruned trees to maximum size

Bagging + random features reduces correlation ρ while maintaining strength s

text

### GBM (R gbm package) [attached_file:1]
Uses deterministic full feature search (max_features=None)

Shallow trees (interaction.depth=1-3)

Fits residuals as regression targets

text

## ❌ ABSOLUTE FORBIDDEN CHANGES
❌ No extra parameters in init
❌ No changing method signatures
❌ No internal state changes (teammates access Node directly)
❌ No scikit-learn style (exact API contract)
❌ No pruning (Breiman: "trees grown are not pruned")
❌ No categorical handling (assume numeric X)

text

## ✅ SUCCESS CHECKLIST
[ ] Class signature matches exactly
[ ] Node has EXACT 5 attributes
[ ] Both modes: deterministic (all feats) + randomized (subset)
[ ] Vectorized split finding (no feat/threshold loops)
[ ] All 4 stopping conditions
[ ] Auto task detection classification/regression
[ ] Reproducible random_state
[ ] Gini/entropy/mse implemented correctly
[ ] Docstrings on ALL methods

text

## 🧪 UNIT TESTS (include in comments)
Impurity
assert gini([0.5,0.5]) == 0.5
assert mse() ≈ 2.666​​

Task detection
assert is_classification() == True​
assert is_regression([1.2,3.4]) == True

Modes
tree1 = DecisionTree(max_features=None) # GBM: all features
tree2 = DecisionTree(max_features='sqrt') # RF: random subset