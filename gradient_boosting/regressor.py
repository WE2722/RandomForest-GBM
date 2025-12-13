

import numpy as np
import sys
sys.path.insert(0, '..')
from decision_tree import DecisionTree


class GradientBoostingRegressor:
    
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        subsample=1.0,
        loss='squared_error',
        random_state=None
    ):
        if loss not in ['squared_error', 'absolute_error']:
            raise ValueError(f"loss must be 'squared_error' or 'absolute_error', got {loss}")
        
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
    
    def _init_prediction(self, y):
      
        if self.loss == 'squared_error':
            return np.mean(y)
        else:  # absolute_error
            return np.median(y)
    
    def _compute_residuals(self, y, y_pred):
       
        diff = y - y_pred
        if self.loss == 'squared_error':
            return diff
        else:  # absolute_error
            return np.sign(diff)
    
    def fit(self, X, y):
       
        X = np.asarray(X)
        y = np.asarray(y)
        self.n_features_ = X.shape[1]
        
       
        if self.random_state is not None:
            rng = np.random.RandomState(self.random_state)
        else:
            rng = np.random.RandomState()
        
       
        self.init_ = self._init_prediction(y)
        y_pred = np.full(X.shape[0], self.init_, dtype=float)
        
    
        self.feature_importances_ = np.zeros(self.n_features_)
   
        self.estimators_ = []
        self.train_score_ = []
        
        
        for m in range(self.n_estimators):
         
            residuals = self._compute_residuals(y, y_pred)
            
           
            if self.subsample < 1.0:
                n_samples = X.shape[0]
                indices = rng.choice(n_samples, size=int(n_samples * self.subsample), replace=False)
                X_fit = X[indices]
                residuals_fit = residuals[indices]
            else:
                X_fit = X
                residuals_fit = residuals
            
       
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
            
            
            loss = np.mean((y - y_pred) ** 2)
            self.train_score_.append(loss)
        
        
        return self
    
    def predict(self, X):

        X = np.asarray(X)
        y_pred = np.full(X.shape[0], self.init_, dtype=float)
        
        for tree in self.estimators_:
            y_pred += self.learning_rate * tree.predict(X)
        
        return y_pred
    
    def staged_predict(self, X):

        X = np.asarray(X)
        y_pred = np.full(X.shape[0], self.init_, dtype=float)
        
        for tree in self.estimators_:
            y_pred += self.learning_rate * tree.predict(X)
            yield y_pred.copy()
