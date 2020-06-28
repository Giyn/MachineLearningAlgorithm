import numpy as np
from .metrics import r2_score


class LinearRegression:

    
    def __init__(self):
        """初始化线性回归模型"""
        self.coef_ = None
        self.intercept_ = None
        self._theta = None
        
        
    def fit_normal(self, X_train, y_train):
        """训练线性回归模型"""
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    
    def predict(self, X_predict):
        """预测的结果向量"""
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)

    
    def score(self, X_test, y_test):
        """模型准确度"""
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    
    def __repr__(self):
        return "LinearRegression()"