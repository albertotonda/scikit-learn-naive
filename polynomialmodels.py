# Wrapper class that implements a regressor with polynoms up to an arbitrary degree, using scikit-learn functions
# Now also including a LogisticRegression variant for classification
# by Alberto Tonda, 2019 <alberto.tonda@gmail.com>

import logging

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.preprocessing import StandardScaler

class PolynomialRegressor :
	
    # attributes
    max_degree = 2
    model = None
    poly = None
    
    def __init__(self, max_degree) :
        self.max_degree = max_degree
        self.model = LinearRegression()
        self.poly = PolynomialFeatures(self.max_degree)

    def __str__(self) :
        return "PolynomialRegressor_%d" % self.max_degree

    def __repr__(self) :
        return "PolynomialRegressor(max_degree=%d)" % self.max_degree
    
    def fit(self, X, y) :
            
        # create new matrix with extra columns
        logging.debug("Creating new matrix...")
        X_extended = self.poly.fit_transform(X) 
        logging.debug("Matrix: %s" % str(X.shape))
        logging.debug("Matrix, extended: %s" % str(X_extended.shape))
        
        # fit a linear model on the new matrix
        logging.debug("Fitting model...")
        self.model.fit(X_extended, y)
        logging.debug("Done!")
        
        return
    
    def predict(self, X) :
        # TODO check if model has been trained; flag? ask LinearRegression?
        X_extended = self.poly.transform(X)
        return self.model.predict(X_extended)

class PolynomialLogisticRegression :

    # attributes
    max_degree = 2
    model = None
    poly = None
    normalize = True
    scaler = StandardScaler()

    def __init__(self, max_degree, normalize=True) :
        self.max_degree = max_degree
        self.model = LogisticRegression()
        self.poly = PolynomialFeatures(self.max_degree)
        self.normalize = normalize

    def __str__(self) :
        return "PolynomialLogisticRegression_%d" % self.max_degree

    def fit(self, X, y) :

        # create new matrix with extra columns
        logging.debug("Creating new matrix...")
        X_extended = self.poly.fit_transform(X) 
        logging.debug("Matrix: %s" % str(X.shape))
        logging.debug("Matrix, extended: %s" % str(X_extended.shape))

        # LogisticRegression usually works best with normalized 
        X_extended_normalized = self.scaler.fit_transform(X_extended)

        # fit a linear model on the new matrix
        logging.debug("Fitting model...")
        self.model.fit(X_extended_normalized, y)
        logging.debug("Done!")
        
        return

    def predict(self, X) :
        # TODO check if model has been trained; flag? ask LogisticRegression?
        X_extended_normalized = self.scaler.transform( self.poly.transform(X) )
        return self.model.predict(X_extended_normalized)
