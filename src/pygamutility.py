"""
Simple script to implement a scikit-learn compatible PyGAMREgressor and a PyGAMClassifier with number of splines depending on number of features
"""
import numpy as np

from pygam import LinearGAM, s 
from sklearn.base import BaseEstimator, RegressorMixin

class PyGAMRegressor(BaseEstimator, RegressorMixin) :

    def fit(self, X, y) :

        # initialize regressor with number of splines equal to number of features
        self._gam = LinearGAM(n_splines=X.shape[1]).fit(X, y)

        return self

    def predict(self, X) :

        return self._gam.predict(X)
