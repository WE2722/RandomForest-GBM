

import numpy as np
import sys
sys.path.insert(0, '..')
from decision_tree import DecisionTree


class GradientBoostingClassifier:
  
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        subsample=1.0,
        loss='log_loss',
        random_state=None
    ):
        if loss != 'log_loss':
            raise ValueError(f"loss must be 'log_loss' for classification, got {loss}")
        
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.loss = loss
        self.random_state = random_state
        
        self.estimators_ = []
        self.init_ = None
        self.feature_importances_ = None
        self.train_score_ = []
        self.n_features_ = None
        self.classes_ = None
        self.n_classes_ = None
    
    def _init_prediction(self, y):
        if self.n_classes_ == 2:
            # Binary classification: log-odds of positive class
            p = np.mean(y == self.classes_[1])
            p = np.clip(p, 1e-10, 1 - 1e-10)
            return np.log(p / (1 - p))
        else:
            # Multiclass: log of class probabilities
            counts = np.array([np.sum(y == c) for c in self.classes_])
            proportions = counts / len(y)
            proportions = np.clip(proportions, 1e-10, 1 - 1e-10)
            return np.log(proportions)
    
    def _compute_residuals(self, y, y_pred):
        if self.n_classes_ == 2:
            # Binary classification
            proba = 1 / (1 + np.exp(-y_pred))
            y_binary = (y == self.classes_[1]).astype(float)
            return y_binary - proba
        else:
            # Multiclass classification
            y_indices = np.searchsorted(self.classes_, y)
            y_one_hot = np.eye(self.n_classes_)[y_indices]
            
        
            exp_pred = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
            softmax = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
            
            return y_one_hot - softmax
    
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.n_features_ = X.shape[1]
        
        # Store classes and detect multiclass
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        if self.random_state is not None:
            rng = np.random.RandomState(self.random_state)
        else:
            rng = np.random.RandomState()
        
        self.init_ = self._init_prediction(y)
        
        if self.n_classes_ == 2:
            y_pred = np.full(len(y), self.init_, dtype=float)
        else:
            y_pred = np.zeros((len(y), self.n_classes_))
            y_pred[:] = self.init_
        
        self.feature_importances_ = np.zeros(self.n_features_)
        
        self.estimators_ = []
        self.train_score_ = []
        
        for m in range(self.n_estimators):
            residuals = self._compute_residuals(y, y_pred)
            
            if self.subsample < 1.0:
                n_samples = len(y)
                indices = rng.choice(n_samples, size=int(n_samples * self.subsample), replace=False)
                X_fit = X[indices]
                residuals_fit = residuals[indices]
            else:
                X_fit = X
                residuals_fit = residuals
            
            if self.n_classes_ == 2:
                tree = DecisionTree(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    criterion='mse',
                    max_features=None,
                    random_state=self.random_state
                )
                tree.fit(X_fit, residuals_fit)
                tree_pred = tree.predict(X)
                y_pred = y_pred + self.learning_rate * tree_pred
                
                self.estimators_.append(tree)
                self.feature_importances_ += tree.feature_importances_
            else:
                trees_list = []
                for k in range(self.n_classes_):
                    tree = DecisionTree(
                        max_depth=self.max_depth,
                        min_samples_split=self.min_samples_split,
                        min_samples_leaf=self.min_samples_leaf,
                        criterion='mse',
                        max_features=None,
                        random_state=self.random_state
                    )
                    tree.fit(X_fit, residuals_fit[:, k])
                    tree_pred = tree.predict(X)
                    y_pred[:, k] = y_pred[:, k] + self.learning_rate * tree_pred
                    trees_list.append(tree)
                    self.feature_importances_ += tree.feature_importances_
                
                self.estimators_.append(trees_list)
            
            if self.n_classes_ == 2:
                proba_train = 1 / (1 + np.exp(-y_pred))
                y_binary = (y == self.classes_[1]).astype(float)
                loss = -np.mean(y_binary * np.log(np.clip(proba_train, 1e-15, 1)) + 
                               (1 - y_binary) * np.log(np.clip(1 - proba_train, 1e-15, 1)))
            else:
                exp_pred = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
                softmax = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
                y_indices = np.searchsorted(self.classes_, y)
                loss = -np.mean(np.log(np.clip(softmax[np.arange(len(y)), y_indices], 1e-15, 1)))
            
            self.train_score_.append(loss)
        
        return self
    
    def predict(self, X):
        X = np.asarray(X)
        
        if self.n_classes_ == 2:
            proba = self.predict_proba(X)
            return self.classes_[(proba[:, 1] >= 0.5).astype(int)]
        else:
            proba = self.predict_proba(X)
            return self.classes_[np.argmax(proba, axis=1)]
    
    def predict_proba(self, X):
        X = np.asarray(X)
        
        if self.n_classes_ == 2:
            raw_pred = np.full(X.shape[0], self.init_, dtype=float)
            
            for tree in self.estimators_:
                raw_pred += self.learning_rate * tree.predict(X)
            
            proba_class1 = 1 / (1 + np.exp(-raw_pred))
            return np.column_stack([1 - proba_class1, proba_class1])
        else:
            raw_pred = np.zeros((X.shape[0], self.n_classes_))
            raw_pred[:] = self.init_
            
            for trees_list in self.estimators_:
                for k, tree in enumerate(trees_list):
                    raw_pred[:, k] += self.learning_rate * tree.predict(X)
            
            # Softmax
            exp_pred = np.exp(raw_pred - np.max(raw_pred, axis=1, keepdims=True))
            return exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
    
    def staged_predict(self, X):
        X = np.asarray(X)
        
        if self.n_classes_ == 2:
            raw_pred = np.full(X.shape[0], self.init_, dtype=float)
            
            for tree in self.estimators_:
                raw_pred += self.learning_rate * tree.predict(X)
                proba = 1 / (1 + np.exp(-raw_pred))
                class_pred = self.classes_[(proba >= 0.5).astype(int)]
                yield class_pred.copy()
        else:
            raw_pred = np.zeros((X.shape[0], self.n_classes_))
            raw_pred[:] = self.init_
            
            for trees_list in self.estimators_:
                for k, tree in enumerate(trees_list):
                    raw_pred[:, k] += self.learning_rate * tree.predict(X)
                
                exp_pred = np.exp(raw_pred - np.max(raw_pred, axis=1, keepdims=True))
                softmax = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
                class_pred = self.classes_[np.argmax(softmax, axis=1)]
                yield class_pred.copy()
    
    def staged_predict_proba(self, X):
        X = np.asarray(X)
        
        if self.n_classes_ == 2:
            raw_pred = np.full(X.shape[0], self.init_, dtype=float)
            
            for tree in self.estimators_:
                raw_pred += self.learning_rate * tree.predict(X)
                proba_class1 = 1 / (1 + np.exp(-raw_pred))
                proba = np.column_stack([1 - proba_class1, proba_class1])
                yield proba.copy()
        else:
            raw_pred = np.zeros((X.shape[0], self.n_classes_))
            raw_pred[:] = self.init_
            
            for trees_list in self.estimators_:
                for k, tree in enumerate(trees_list):
                    raw_pred[:, k] += self.learning_rate * tree.predict(X)
                
                exp_pred = np.exp(raw_pred - np.max(raw_pred, axis=1, keepdims=True))
                softmax = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
                yield softmax.copy()
