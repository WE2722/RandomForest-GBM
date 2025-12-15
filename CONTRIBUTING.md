# Contributing Guidelines

## Team Structure & Folder Ownership

| Member | Role | Folder | GitHub |
|--------|------|--------|--------|
| **Wiame** | Decision Tree (CART) | `decision_tree/` | [@WE2722](https://github.com/WE2722) |
| **Abdellah** | Random Forest | `random_forest/` | [@AbdellahBaqua](https://github.com/AbdellahBaqua) |
| **Saif** | Gradient Boosting | `gradient_boosting/` | [@Saif-dbot](https://github.com/Saif-dbot) |

---

## PROTECTED FOLDERS

Each member's folder is **PROTECTED**. You can **ONLY** modify your own folder!

```
Repository Structure:
decision_tree/      -> PROTECTED - Only Wiame (@WE2722)
random_forest/      -> PROTECTED - Only Abdellah (@AbdellahBaqua)  
gradient_boosting/  -> PROTECTED - Only Saif (@Saif-dbot)
.github/, configs   -> PROTECTED - Only Wiame 
```

### What happens if you modify someone else's folder?

The CI will **FAIL** with this error:

```
PERMISSION VIOLATIONS DETECTED
==============================================
[X] random_forest/ - Only AbdellahBaqua (Abdellah) can modify

FOLDER OWNERSHIP:
  decision_tree/     -> WE2722 (Wiame)
  random_forest/     -> AbdellahBaqua (Abdellah)
  gradient_boosting/ -> Saif-dbot (Saif)
==============================================
```

---

## What You CAN Modify

### Wiame (@WE2722)
- `decision_tree/*`
- `tests/test_classification_tree.py`
- `tests/test_regression_tree.py`
- `tests/my_tree_vs_sklearn.py`
- `.github/*`, `README.md`, `CONTRIBUTING.md`

### Abdellah (@AbdellahBaqua)
- `random_forest/*`
- `random_forest/tests.py` (create this)

### Saif (@Saif-dbot)
- `gradient_boosting/*`
- `gradient_boosting/tests.py` (create this)

---

## Git Workflow

### Step 1: Clone
```bash
git clone https://github.com/WE2722/RandomForest-GBM.git
cd RandomForest-GBM
```

### Step 2: Create Branch
```bash
git checkout main
git pull origin main
git checkout -b feature/your-feature-name
```

### Step 3: Make Changes (YOUR FOLDER ONLY!)
```bash
# Abdellah:
git add random_forest/

# Saif:
git add gradient_boosting/
```

### Step 4: Commit and Push
```bash
git commit -m "feat(rf): implement bootstrap sampling"
git push origin feature/your-feature-name
```

### Step 5: Create Pull Request on GitHub
1. Go to repository
2. Click "Pull requests" -> "New pull request"
3. Wait for CI checks to pass
4. Merge when approved

---

## Commit Messages

Format: `type(scope): description`

```bash
# Wiame
git commit -m "fix(dt): handle edge case"

# Abdellah
git commit -m "feat(rf): implement OOB score"

# Saif
git commit -m "feat(gbm): add learning rate"
```

---

## Need Changes to Another Member's Code?

1. Create an **Issue** on GitHub
2. Tag the folder owner (@WE2722, @AbdellahBaqua, or @Saif-dbot)
3. Describe what change you need
4. Wait for them to make the change

---

## Contact

| Member | GitHub | Folder |
|--------|--------|--------|
| Wiame | [@WE2722](https://github.com/WE2722) | decision_tree/ |
| Abdellah | [@AbdellahBaqua](https://github.com/AbdellahBaqua) | random_forest/ |
| Saif | [@Saif-dbot](https://github.com/Saif-dbot) | gradient_boosting/ |
