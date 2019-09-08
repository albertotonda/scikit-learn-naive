# Simple Python script that iterates over several regression methods and tests them on a given dataset
# by Alberto Tonda, 2016-2019 <alberto.tonda@gmail.com>

import datetime
import logging
import numpy as np
import os
import pickle
import sys

# my home-made stuff
import commonLibrary
from keraswrappers import ANNRegressor
from polynomialregressor import PolynomialRegressor

# scikit-learn stuff
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import LeaveOneOut

from sklearn.preprocessing import StandardScaler

# list of all regressors included in scikit-learn
from sklearn.cross_decomposition import PLSRegression

from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

# these methods do not have the .fit(X,y) method
from sklearn.gaussian_process import regression_models

from sklearn.isotonic import IsotonicRegression

from sklearn.kernel_ridge import KernelRidge

from sklearn.linear_model import ARDRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LarsCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import PassiveAggressiveRegressor

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor

from sklearn.neural_network import BernoulliRBM

from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.svm import NuSVR

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor

# and here is the keras-related stuff
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

# now, this is new and exciting stuff: Generalized Additive Models (GAM) from module pyGAM
from pygam import LinearGAM 

# this is a class that can be used to wrap Eureqa equations
# the annoying part is to find the corresponding named variables inside
# array 'X'
class EureqaRegressor :
	
	def fit(X, y) :

		# basically do nothing, the fitting has been performed offline
		return
	
	def predict(X) :

		# here, we return an np array with the points predicted by the
		# equation that we selected
		y = np.zeros( len(X) )

		for i in range(len(X)) :
			y[i] = 1.234 * X[i][0]
		
		return y

# TODO move relativeFeatureImportance to a common library (?)
def relativeFeatureImportance(classifier) :
	
	# this is the output; it will be a sorted list of tuples (importance, index)
	# the index is going to be used to find the "true name" of the feature
	orderedFeatures = []

	# the simplest case: the classifier already has a method that returns relative importance of features
	if hasattr(classifier, "feature_importances_") :

		orderedFeatures = zip(classifier.feature_importances_ , range(0, len(classifier.feature_importances_)))
		orderedFeatures = sorted(orderedFeatures, key = lambda x : x[0], reverse=True)
	
	# the classifier does not have "feature_importances_" but can return a list
	# of all features used by a lot of estimators (typical of ensembles)
	elif hasattr(classifier, "estimators_features_") :

		numberOfFeaturesUsed = 0
		featureFrequency = dict()
		for listOfFeatures in classifier.estimators_features_ :
			for feature in listOfFeatures :
				if feature in featureFrequency :
					featureFrequency[feature] += 1
				else :
					featureFrequency[feature] = 1
			numberOfFeaturesUsed += len(listOfFeatures)
		
		for feature in featureFrequency : 
			featureFrequency[feature] /= numberOfFeaturesUsed

		# prepare a list of tuples (name, value), to be sorted
		orderedFeatures = [ (featureFrequency[feature], feature) for feature in featureFrequency ]
		orderedFeatures = sorted(orderedFeatures, key=lambda x : x[0], reverse=True)

	# the classifier does not even have the "estimators_features_", but it's
	# some sort of linear/hyperplane classifier, so it does have a list of
	# coefficients; for the coefficients, the absolute value might be relevant
	elif hasattr(classifier, "coef_") :
	
		# now, "coef_" is usually multi-dimensional, so we iterate on
		# all dimensions, and take a look at the features whose coefficients
		# more often appear close to the top; but it could be mono-dimensional,
		# so we need two special cases
		dimensions = len(classifier.coef_.shape)
		#logging.info("dimensions=", len(dimensions))
		featureFrequency = None # to be initialized later
		
		# check on the dimensions
		if dimensions == 1 :
			featureFrequency = np.zeros(len(classifier.coef_))
			
			relativeFeatures = zip(classifier.coef_, range(0, len(classifier.coef_)))
			relativeFeatures = sorted(relativeFeatures, key=lambda x : abs(x[0]), reverse=True)
			
			for index, values in enumerate(relativeFeatures) :
				value, feature = values
				featureFrequency[feature] += 1/(1+index)

		elif dimensions > 1 :
			featureFrequency = np.zeros(len(classifier.coef_[0]))
			
			# so, for each dimension (corresponding to a class, I guess)
			for i in range(0, len(classifier.coef_)) :
				# we give a bonus to the feature proportional to
				# its relative order, good ol' 1/(1+index)
				relativeFeatures = zip(classifier.coef_[i], range(0, len(classifier.coef_[i])))
				relativeFeatures = sorted(relativeFeatures, key=lambda x : abs(x[0]), reverse=True)
				
				for index, values in enumerate(relativeFeatures) :
					value, feature = values
					featureFrequency[feature] += 1/(1+index)
			
		# finally, let's sort
		orderedFeatures = [ (featureFrequency[feature], feature) for feature in range(0, len(featureFrequency)) ]
		orderedFeatures = sorted(orderedFeatures, key=lambda x : x[0], reverse=True)

	else :
		logging.info("The classifier does not have any way to return a list with the relative importance of the features")

	return np.array(orderedFeatures)

########################################## MAIN ###################################################
def main() :

	# let's create a folder with a unique name to store results
	folderName = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + "-regression" 
	if not os.path.exists(folderName) : os.makedirs(folderName)
	
	# initialize logging
	commonLibrary.initialize_logging(folderName)
	
	regressorsList = [
			
			[PolynomialRegressor(2), "PolynomialRegressor2"],
			[PolynomialRegressor(3), "PolynomialRegressor3"],
			# keras neural network
			#[ANNRegressor(epochs=500, batch_size=32, layers=[16,4]), "KerasRegressor8-4"],
			#[ANNRegressor(epochs=700, batch_size=32, layers=[16,8]), "KerasRegressor16-8"],
			
			# cross decomposition
			[PLSRegression(), "PLSRegression"],

			# ensemble
			[AdaBoostRegressor(), "AdaBoostRegressor"],
			[BaggingRegressor(), "BaggingRegressor"],
			[BaggingRegressor(n_estimators=100), "BaggingRegressor_100"],
			[BaggingRegressor(n_estimators=300), "BaggingRegressor_300"],
			[ExtraTreesRegressor(), "ExtraTreesRegressor"],
			[GradientBoostingRegressor(), "GradientBoostingRegressor"],
			[RandomForestRegressor(), "RandomForestRegressor"],
			[RandomForestRegressor(n_estimators=100), "RandomForestRegressor_100"],
			[RandomForestRegressor(n_estimators=300), "RandomForestRegressor_300"],
			
			# isotonic
			#[IsotonicRegression(), "IsotonicRegression"], # apparently wants "X" as a 1d array
			
			# kernel ridge
			[KernelRidge(), "KernelRidge"],

			# linear
			#[ARDRegression(), "ARDRegression"], # takes too much time to train
			[BayesianRidge(), "BayesianRidge"],
			[ElasticNetCV(), "ElasticNetCV"],
			[LarsCV(), "LarsCV"],
			[LassoCV(), "LassoCV"],
			[LinearRegression(), "LinearRegression"],
			[PassiveAggressiveRegressor(), "PassiveAggressiveRegressor"],
		
			# neighbors
			[KNeighborsRegressor(), "KNeighborsRegressor"],
			[RadiusNeighborsRegressor(), "RadiusNeighborsRegressor"],
			
			# neural networks
			#[BernoulliRBM(), "BernoulliRBM"], # has a different interface, no "predict"
			
			# svm
			[SVR(), "SVR"],
			[LinearSVR(), "LinearSVR"],
			[NuSVR(), "NuSVR"],
			
			# tree
			[DecisionTreeRegressor(), "DecisionTreeRegressor (max depth 10)"],
			[ExtraTreeRegressor(), "ExtraTreeRegressor"],
			
			# generalized additive models
			[LinearGAM(n_splines=20), "LinearGAM(n_splines=20)"],
			
			]

	X = y = X_train = X_test = y_train = y_test = variablesX = variablesY = None
	numberOfSplits = 10 # TODO change number of splits from command line
	
	if True :
		X, y, variablesX, variablesY = commonLibrary.loadYongShiDataCalibration2("TIMBER")

	if False :
		X, y, variablesX, variablesY = commonLibrary.loadLaurentBouvierNewData()
	
	if False :
		X, y, variablesX, variablesY = commonLibrary.loadYongShiDataCalibration()
	
	if False :
		from sklearn.datasets import load_linnerud
		X, y = load_linnerud(return_X_y=True)
	
	if False :
		X, y, variablesX, variablesY = commonLibrary.loadYingYingData()

	if False :
		X, y, variablesX, variablesY = commonLibrary.loadCleaningDataGermanSpecific()
		#X, y, variablesX, variablesY = commonLibrary.loadCleaningDataGerman()

	if False :
		X, y, variablesX, variablesY = commonLibrary.loadInsects()

	if False :
		X, y, variablesX, variablesY = commonLibrary.loadMilkProcessPipesDimensionalAnalysis()
		#X, y, variablesX, variablesY = commonLibrary.loadMilkProcessPipes()

	if False : # ecosystem services
		X, y, variablesX, variablesY = commonLibrary.loadEcosystemServices()
	
	if False :
		X, y, variablesX, variablesY = commonLibrary.loadMarcoSoil()
	
	if False : 
		# load dataset
		X, y = commonLibrary.loadEureqaRegression()
		# randomly split between training and test
		#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

	if False :
		# load dataset
		X_train, X_test, y_train, y_test = commonLibrary.loadBiscuitExample()
		logging.info("X_train: " + str(X_train.shape))
		logging.info("X_test: "  + str(X_test.shape))
		logging.info("y_train: " + str(y_train.shape))
		logging.info("y_test: "  + str(y_test.shape))
		
		# in this particular case, I create the "global" X and y by putting together the two arrays
		X = np.append(X_train, X_test, axis=0)
		y = np.append(y_train, y_test, axis=0)

	if False : 
		# load dataset
		X_train, X_test, y_train, y_test = commonLibrary.loadAromoptiExample()
		logging.info("X_train: " + str(X_train.shape))
		logging.info("X_test: "  + str(X_test.shape))
		logging.info("y_train: " + str(y_train.shape))
		logging.info("y_test: "  + str(y_test.shape))
		
		# in this particular case, I create the "global" X and y by putting together the two arrays
		X = np.append(X_train, X_test, axis=0)
		y = np.append(y_train, y_test, axis=0)
	
	logging.info("Regressing %d output variables, in function of %d input variables..." % (len(y[0]), len(X[0])))
	
	# if the names of the variables are not specified, let's specify them!
	if variablesY == None : variablesY = [ "y" + str(i) for i in range(0, len(y[0])) ]
	if variablesX == None : variablesX = [ "X" + str(i) for i in range(0, len(X[0])) ]

	performances = dict()

	for variableIndex, variableY in enumerate(variablesY) :

		logging.info("** Now evaluating models for variable \"%s\"... **" % variableY)

		# obtain data
		y_ = y[:,variableIndex].ravel()

		# assume here that you will have train/test indexes instead
		# it's also easier for the plots, as we do not face the issue
		# of duplicate values (e.g. same value with two indexes)
		rs = ShuffleSplit(n_splits=numberOfSplits, random_state=42)
		#rs = LeaveOneOut()

		# initialize performance dictionary of arrays
		performances[variableY] = dict()
		for regressor, regressorName in regressorsList : 
			performances[variableY][regressorName] = dict()
			performances[variableY][regressorName]["r^2"] = []
			performances[variableY][regressorName]["e.v"] = []
			performances[variableY][regressorName]["mse"] = []
			performances[variableY][regressorName]["mae"] = []
			performances[variableY][regressorName]["predicted"] = []
			
		# this is used to store all values of each fold, in order; maybe there's a smarter way to do it
		foldPointsInOrder = []
			
		# and now, for every regressor
		for foldIndex, indexes in enumerate(rs.split(X)) :

			train_index, test_index = indexes

			X_train = X[train_index]
			y_train = y_[train_index]
			X_test = X[test_index]
			y_test = y_[test_index]
			
			# normalize
			logging.info("Normalizing data...")
			scalerX = StandardScaler()
			scalerY = StandardScaler()

			X_train = scalerX.fit_transform(X_train)
			X_test = scalerX.transform(X_test)
			
			y_train = scalerY.fit_transform(y_train.reshape(-1,1)).ravel() # this "reshape/ravel" here is just to avoid warnings, it has no true effect on data
			y_test = scalerY.transform(y_test.reshape(-1,1)).ravel()
			
			# now, we store points of the folder in order of how they appear
			foldPointsInOrder.extend( list(scalerY.inverse_transform(y_test)) )
			
			for regressorIndex, regressorData in enumerate(regressorsList) :
				
				regressor = regressorData[0]
				regressorName = regressorData[1]
			
				logging.info( "Fold #%d/%d: training regressor #%d/%d \"%s\"" % (foldIndex+1, numberOfSplits, regressorIndex+1, len(regressorsList), regressorName) )

				try :
					regressor.fit(X_train, y_train)
					
					y_test_predicted = regressor.predict(X_test)
					r2Test = r2_score(y_test, y_test_predicted)
					mseTest = mean_squared_error(y_test, y_test_predicted)
					maeTest = mean_absolute_error(y_test, y_test_predicted)
					varianceTest = explained_variance_score(y_test, y_test_predicted)

					logging.info("R^2 score (test): %.4f" % r2Test)
					logging.info("EV score (test): %.4f" % varianceTest)
					logging.info("MSE score (test): %.4f" % mseTest)
					logging.info("MAE score (test): %.4f" % maeTest)
					
					# add performance to the list of performances
					performances[variableY][regressorName]["r^2"].append( r2Test )
					performances[variableY][regressorName]["e.v"].append( varianceTest )
					performances[variableY][regressorName]["mse"].append( mseTest )
					performances[variableY][regressorName]["mae"].append( maeTest )
					# also record the predictions, to be used later in a global figure
					performances[variableY][regressorName]["predicted"].extend( list(scalerY.inverse_transform(y_test_predicted)) )
					
					try: 
						import matplotlib.pyplot as plt

						# plotting first figure, with points 'x' and 'o'
						y_predicted = regressor.predict(scalerX.transform(X)) # 'X' was never wholly rescaled before
						y_train_predicted = regressor.predict(X_train)
						
						plt.figure()
						
						plt.scatter(train_index, y_train, c="gray", label="training data")
						plt.scatter(test_index, y_test, c="green", label="test data")

						plt.plot(np.arange(len(y_predicted)), y_predicted, 'x', c="red", label="regression")
						plt.xlabel("order of data samples")
						plt.ylabel("target")
						plt.title(regressorName + ", R^2=%.4f (test)" % r2Test)
						plt.legend()
						
						logging.info("Saving figure...")
						plt.savefig( os.path.join(folderName, regressorName + "-" + variableY + "-fold-" + str(foldIndex+1) + ".pdf") )
						plt.close()
						
						# plotting second figure, with everything close to a middle line
						plt.figure()
						
						plt.plot(y_train, y_train_predicted, 'r.', label="training set") # points
						plt.plot(y_test, y_test_predicted, 'go', label="test set") # points
						plt.plot([min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())], [min(y_train_predicted.min(), y_test_predicted.min()), max(y_train_predicted.max(), y_test_predicted.max())], 'k--') # line
						
						plt.xlabel("measured")
						plt.ylabel("predicted")
						plt.title(regressorName + " measured vs predicted, " + variableY)
						plt.legend(loc='best')

						plt.savefig( os.path.join(folderName, regressorName + "-" + variableY + "-fold-" + str(foldIndex+1) + "-b.pdf") )
						plt.close()
						
						# also, save ordered list of features
						featuresByImportance = relativeFeatureImportance(regressor)

						# if list exists, write feature importance to disk
						# TODO horrible hack here, to avoid issues with GAM
						if len(featuresByImportance) > 0 and "GAM" not in regressorName :
							featureImportanceFileName = regressorName + "-" + variableY + "-featureImportance-fold" + str(foldIndex) + ".csv"
							with open( os.path.join(folderName, featureImportanceFileName), "w") as fp :
								fp.write("feature,importance\n")
								for featureImportance, featureIndex in featuresByImportance :
									fp.write( variablesX[int(featureIndex)] + "," + str(featureImportance) + "\n")
							

					except ImportError:
						logging.info("Cannot import matplotlib. Skipping plots...")
				
				except Exception as e:
					logging.info("Regressor \"" + regressorName + "\" failed on variable \"" + variableY + "\":" + str(e))

	logging.info("Final summary:")
	with open( os.path.join(folderName, "00_summary.txt"), "w") as fp :
			
		for variableY in variablesY :

			logging.info("For variable \"" + variableY + "\"")
			fp.write("For variable: " + variableY + " = f(" + variablesX[0])
			for i in range(1, len(variablesX)) : fp.write("," + variablesX[i])
			fp.write(")\n")
			
			# create a list from the dictionary and sort it
			sortedPerformances = sorted( [ (performances[variableY][regressorName], regressorName) for regressorName in performances[variableY] ], key=lambda x : np.mean( x[0]["r^2"] ), reverse=True)

			for regressorData in sortedPerformances :
				regressorName = regressorData[1]
				regressorScore = regressorData[0]
				
				r2Mean = np.mean( regressorScore["r^2"] )
				r2std = np.std( regressorScore["r^2"] )

				varianceMean = np.mean( regressorScore["e.v"] )
				varianceStd = np.std( regressorScore["e.v"] )

				mseMean = np.mean( regressorScore["mse"] )
				mseStd = np.std( regressorScore["mse"] )

				maeMean = np.mean( regressorScore["mae"] )
				maeStd = np.std( regressorScore["mae"] )

				logging.info("\t- %s, R^2=%.4f (std=%.4f), Explained Variance=%.4f (std=%.4f), MSE=%.4f (std=%.4f), MAE=%.4f (std=%.4f)" % 
					(regressorName, r2Mean, r2std, varianceMean, varianceStd, mseMean, mseStd, maeMean, maeStd) )

				fp.write("\t- %s, R^2=%.4f (std=%.4f), Explained Variance=%.4f (std=%.4f), MSE=%.4f (std=%.4f), MAE=%.4f (std=%.4f)\n" % 
					(regressorName, r2Mean, r2std, varianceMean, varianceStd, mseMean, mseStd, maeMean, maeStd))

				fp.write("\t\t- R^2:" + str([ "%.4f" % x for x in regressorScore["r^2"] ]) + "\n") 
				fp.write("\t\t- E.V.:" + str([ "%.4f" % x for x in regressorScore["e.v"] ]) + "\n") 
				fp.write("\t\t- MSE:" + str([ "%.4f" % x for x in regressorScore["mse"] ]) + "\n") 
				fp.write("\t\t- MAE:" + str([ "%.4f" % x for x in regressorScore["mae"] ]) + "\n") 
				
				# also, plot a "global" graph
				# issue here, if a regressor fails, you haveincongruent matrixes: a check is in order
				if len(foldPointsInOrder) == len( regressorScore["predicted"] ) :
					fig = plt.figure()
					ax = fig.add_subplot(111)
					
					#bottom_left_corner = [min(foldPointsInOrder), max(foldPointsInOrder)]
					#top_right_corner = [min(regressorScore["predicted"]), max(regressorScore["predicted"])]
					x_bottom_top = [0, max(foldPointsInOrder)]
					y_bottom_top = [0, max(foldPointsInOrder)]

					ax.plot(foldPointsInOrder, regressorScore["predicted"], 'g.') # points
					ax.plot(x_bottom_top, y_bottom_top, 'k--', label="1:1") # line
					ax.plot(x_bottom_top, [y_bottom_top[0]*1.20, y_bottom_top[1]*1.20], 'r--', label="20% error")
					ax.plot(x_bottom_top, [y_bottom_top[0]*0.80, y_bottom_top[1]*0.80], 'r--')
					
					ax.set_title(regressorName + " measured vs predicted, " + variableY + " (all test)")
					ax.set_xlabel("measured")
					ax.set_ylabel("predicted")
					ax.legend(loc='best')
		
					plt.savefig( os.path.join(folderName, regressorName + "-" + variableY + "-global-b.png") )
					plt.close(fig)

	#logging.info("Saving the best regressor for each variable:")
	#for i in range(0, iterations) :
	#	
	#	fileName = os.path.join(folderName, "best-regressor-y" + str(i) + ".pickle")
	#	
	#	# this part is a little bit weird, probably changing the original list would be better, but who cares?
	#	sortedPerformances = sorted( performances[i], key=lambda x : x[0], reverse=True)
	#	j = 0
	#	while regressorsList[j][1] != sortedPerformances[0][1] : j += 1
	#	
	#	logging.info("For variable y" + str(i) + ", pickleing \"" + regressorsList[j][1] + "\" as " + fileName + "...") 
	#	pickle.dump( regressorsList[j][0], open(fileName, "wb"))

# stuff to make the script more proper
if __name__ == "__main__" :
	sys.exit(main())
