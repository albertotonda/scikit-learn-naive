# Simple script to test the best classifier for the problem
# by Alberto Tonda, 2016-2022 <alberto.tonda@gmail.com>

import copy
import datetime
import itertools
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re as regex
import sys

# TODO this is just used to suppress all annoying warnings from scikit-learn, but it's not great
import warnings
warnings.filterwarnings("ignore")

# this is to get the parameters in a function
from inspect import signature, Parameter

# local libraries
import common
from polynomialmodels import PolynomialLogisticRegression 
from keraswrappers import ANNClassifier

# here are some utility functions, for cross-validation, scaling, evaluation, et similia
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score, roc_curve, RocCurveDisplay 
from sklearn.utils import all_estimators

# here are all the classifiers
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import SGDClassifier

from sklearn.multiclass import OneVsOneClassifier 
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OutputCodeClassifier

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import RadiusNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier

# TODO: divide classifiers into "families" and test each family separately
# TODO: there are some very complex classifiers, such as VotingClassifier: to be explored
# TODO: also, GridSearchCv makes an exhaustive search over the classifer's parameters (?): to be explored
# TODO: refactor code properly, add command-line options
# TODO: take all classifiers that are currently not working, and try to wrap them in a class with .fit() and .predict() and .score()
# TODO: the same, but with pyGAM (Generalized Additive Models)

# NOTE: comment/uncomment classifiers from the list
classifier_list = [

    # ensemble
    AdaBoostClassifier(),
    BaggingClassifier(),
    BaggingClassifier(n_estimators=300),
    ExtraTreesClassifier(),
    GradientBoostingClassifier(),
    GradientBoostingClassifier(n_estimators=300),
    RandomForestClassifier(),
    RandomForestClassifier(n_estimators=300, class_weight='balanced'),

    # linear
    LogisticRegression(),
    LogisticRegressionCV(),
    PassiveAggressiveClassifier(),
    RidgeClassifier(),
    RidgeClassifierCV(),
    SGDClassifier(),
    SVC(kernel='linear'),
    
    # naive Bayes
    BernoulliNB(),
    GaussianNB(),
    #MultinomialNB(),
    
    # neighbors
    KNeighborsClassifier(),
    # TODO this one creates issues
    #NearestCentroid(), # it does not have some necessary methods, apparently
    #RadiusNeighborsClassifier(),
    
    # tree
    DecisionTreeClassifier(),
    ExtraTreeClassifier(),

    # polynomial NOTE for large datasets (e.g. lots of features) this will explode
    #PolynomialLogisticRegression(max_degree=2),

    # neural networks NOTE they take a long time to train, parameters have to be tweaked
    #ANNClassifier(layers=[8,4])

    ]

ensemble_classifier_list = [

    OneVsOneClassifier,
    OneVsRestClassifier,
    OutputCodeClassifier

    ]

# this function plots a confusion matrix
def plot_confusion_matrix(confusionMatrix, classes, fileName, title='Confusion matrix'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cmNormalized = confusionMatrix.astype('float') / confusionMatrix.sum(axis=1)[:, np.newaxis]

    # attempt at creating a more meaningful visualization: values that are in the "wrong box"
    # (for which predicted label is different than true label) are turned negative
    for i in range(0, cmNormalized.shape[0]) :
        for j in range(0, cmNormalized.shape[1]) :
            if i != j : cmNormalized[i,j] *= -1.0

    fig = plt.figure()
    plt.imshow(cmNormalized, vmin=-1.0, vmax=1.0, interpolation='nearest', cmap='RdBu') #cmap='RdYlGn')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cmNormalized.max() / 2.
    for i, j in itertools.product(range(cmNormalized.shape[0]), range(cmNormalized.shape[1])):
        text = "%.2f\n(%d)" % (abs(cmNormalized[i,j]), confusionMatrix[i,j])
        plt.text(j, i, text, horizontalalignment="center", color="white" if cmNormalized[i,j] > thresh or cmNormalized[i,j] < -thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    fig.subplots_adjust(bottom=0.2)
    plt.savefig(fileName)
    plt.close()

    return

# this function returns a list of features, in relative order of importance
def get_relative_feature_importance(classifier) :
	
    # this is the output; it will be a sorted list of tuples (importance, index)
    # the index is going to be used to find the "true name" of the feature
    orderedFeatures = []

    # the simplest case: the classifier already has a method that returns relative importance of features
    if hasattr(classifier, "feature_importances_") :

        orderedFeatures = zip(classifier.feature_importances_ , range(0, len(classifier.feature_importances_)))
        orderedFeatures = sorted(orderedFeatures, key = lambda x : x[0], reverse=True)

    # some classifiers are ensembles, and if each element in the ensemble is able to return a list of feature importances
    # (that are going to be all scored following the same logic, so they could be easily aggregated, in theory)
    elif hasattr(classifier, "estimators_") and hasattr(classifier.estimators_[0], "feature_importances_") :

        # add up the scores given by each estimator to all features
        global_score = np.zeros(classifier.estimators_[0].feature_importances_.shape[0])

        for estimator in classifier.estimators_ :
            for i in range(0, estimator.feature_importances_.shape[0]) :
                global_score[i] += estimator.feature_importances_[i]

        # "normalize", dividing by the number of estimators
        for i in range(0, global_score.shape[0]) : global_score[i] /= len(classifier.estimators_)

        # proceed as above to obtain the ranked list of features
        orderedFeatures = zip(global_score, range(0, len(global_score)))
        orderedFeatures = sorted(orderedFeatures, key = lambda x : x[0], reverse=True)
	
    # the classifier does not have "feature_importances_" but can return a list
    # of all features used by a lot of estimators (typical of ensembles)
    # TODO this case below DOES NOT WORK for Bagging, where "feature_importances_" has another meaning
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
        #print("dimensions=", len(dimensions))
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
        logging.warning("The classifier does not have any way to return a list with the relative importance of the features")

    return np.array(orderedFeatures)

############################################################## MAIN
def main() :
	
    # TODO argparse? maybe divide into "fast", "exhaustive", "heuristic"; also add option to specify file from command line (?)
    # hard-coded values here
    n_splits = 10
    final_report_file_name = "00_final_report.txt"

    # this is a dictionary of pointers to functions, used for all classification metrics that accept
    # (y_true, y_pred) as positional arguments
    metrics = {}
    metrics["accuracy"] = accuracy_score
    metrics["F1"] = f1_score
    metrics["MCC"] = matthews_corrcoef

    # create uniquely named folder
    folder_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + "-classification" 
    if not os.path.exists(folder_name) : os.makedirs(folder_name)
    
    # start logging
    common.initialize_logging(folder_name)

    # generate a random seed that will be used for all the experiments
    random_seed = int(datetime.datetime.now().timestamp())
    logging.info("Random seed that will be used for all experiments: %d" % random_seed)

    # generate the list of classifiers
    classifier_list = []
    estimators = all_estimators(type_filter="classifier")

    # TODO before entering this loop, we could actually add extra classifiers from other sources, as long as they are scikit-learn compatible
    for name, class_ in estimators :
        # try to infer if classifiers accept special parameters (e.g. 'random_seed') and add them as keyword arguments
        # we try to instantiate the classifier
        try :
            # first, get all the parameters in the initialization function
            sig = signature(class_.__init__)
            params = sig.parameters # these are not regular parameters, yet

            # we need to convert them to a dictionary
            params_dict = {}
            for p_name, param in params.items() :
                if params[p_name].default != Parameter.empty :
                    params_dict[p_name] = params[p_name].default

            # if the classifier requires a random seed, set it
            if 'random_seed' in params :
                params_dict['random_seed'] = random_seed

            classifier_list.append( class_(**params_dict) )

            # if it accepts a parameter called 'n_estimators', let's create a second instance and go overboard 
            if 'n_estimators' in params :
                params_dict['n_estimators'] = 300
                classifier_list.append( class_(**params_dict) )

        except Exception as e :
            logging.error("Cannot instantiate classifier \"%s\" (exception: \"%s\"), skipping..." % (name, str(e))) 

    # TODO BRUTALLY REMOVE ALL CLASSIFIERS EXCEPT RANDOM FOREST, JUST TO SPEED UP TESTING (this part has to be changed)
    classifier_list = [ c for c in classifier_list if str(c).startswith("RandomForest") ]

    logging.info("A total of %d classifiers will be used: %s" % (len(classifier_list), str(classifier_list)))
    
    # this part can be used by some case studies, storing variable names
    variableY = variablesX = None

    # get data
    logging.info("Loading data...")
    #X, y, variablesX, variablesY = common.loadRallouData() # TODO replace here to load different data
    #X, y, variablesX, variablesY = common.loadCoronaData()
    #X, y, variablesX, variablesY = common.loadXORData()
    #X, y, variablesX, variablesY = common.loadMl4Microbiome()
    X, y, variablesX, variablesY = common.loadMl4MicrobiomeCRC()
    variableY = variablesY[0]

    logging.info("Shape of X: " + str(X.shape))
    logging.info("Shape of y: " + str(y.shape))
    
    # also take note of the classes, they will be useful in the following
    classes, classesCount = np.unique(y, return_counts=True)
	
    # let's output some details about the data, that might be important
    logging.info("Class distribution for the %d classes." % len(classes))
    for i, c in enumerate(classes) :
        logging.info("- Class %d has %.4f of the samples in the dataset." % (c, float(classesCount[i]) / float(y.shape[0])))
	
    # an interesting comparison: what's the performance of a random classifier?
    random_scores = { metric_name : [] for metric_name, metric_function in metrics.items() }
    for i in range(0, 100) :
        y_random = np.random.randint( min(classes), high=max(classes)+1, size=y.shape[0] )
        for metric_name, metric_function in metrics.items() :
            random_scores[metric_name].append( metric_function(y, y_random) )
    logging.info("As a comparison, randomly picking labels 100 times returns the following scores:")
    for metric_name, metric_scores in random_scores.items() :
        logging.info("- Mean %s: %.4f (+/- %.4f)" % (metric_name, np.mean(metric_scores), np.std(metric_scores)))

    # check: do the variables' names exist? if not, put some placeholders
    if variableY is None : variableY = "Y"
    if variablesX is None : variablesX = [ "X" + str(i) for i in range(0, X.shape[1]) ]
	
    # this is a utility dictionary, that will be used to create a more concise summary
    performances = dict()

    # perform stratified k-fold cross-validation, but explicitly
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    folds = [ [train_index, test_index] for train_index, test_index in skf.split(X, y) ]
	
    # TODO 	
    # - also call function for feature selection
    # - also keep track of time needed for each classification
    for classifierIndex, classifierOriginal in enumerate(classifier_list) :
		
        classifier = copy.deepcopy( classifierOriginal )
        classifier_string = str(classifier)

        # now, we automatically generate the name of the classifier, using a regular expression
        classifierName = classifier_string.split("(")[0]
        match = regex.search("n_estimators=([0-9]+)", classifier_string)
        if match : classifierName += "_" + match.group(1)

        logging.info("Classifier #%d/%d: %s..." % (classifierIndex+1, len(classifier_list), classifierName))
        
        # initialize local performance
        performances[classifierName] = dict()
        
        # vector that contains (at the moment) two possibilities
        dataPreprocessingOptions = ["raw", "normalized"]
		
        for dataPreprocessing in dataPreprocessingOptions :

            # create list
            performances[classifierName][dataPreprocessing] = []
			
            # this is used to produce a "global" confusion matrix for the classifier
            all_y_test = []
            all_y_pred = []

            # iterate over all splits
            splitIndex = 0
            for train_index, test_index in folds :

                X_train, X_test = X[train_index], X[test_index] 
                y_train, y_test = y[train_index], y[test_index] 
				
                if dataPreprocessing == "normalized" :
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
			
                logging.info("Training classifier %s on split #%d/%d (%s data)..." % (classifierName, splitIndex+1, n_splits, dataPreprocessing))
                try:
                    classifier.fit(X_train, y_train)
					
                    # instead of calling the classifier's "score" method, let's compute all metrics explicitly
                    y_train_pred = classifier.predict(X_train)
                    y_test_pred = classifier.predict(X_test)
					
                    trainScore = { metric_name : metric_function(y_train, y_train_pred) for metric_name, metric_function in metrics.items() }
                    testScore = { metric_name : metric_function(y_test, y_test_pred) for metric_name, metric_function in metrics.items() }

                    for metric_name in trainScore :
                        logging.info("- %s: Training score: %.4f ; Test score: %.4f" % (metric_name, trainScore[metric_name], testScore[metric_name]))
					
                    # store performance and information
                    performances[classifierName][dataPreprocessing].append( (testScore, trainScore) )
                    all_y_test = np.append(all_y_test, y_test)
                    all_y_pred = np.append(all_y_pred, y_test_pred)
					
                    # get features, ordered by importance 
                    featuresByImportance = get_relative_feature_importance(classifier)
					
                    # write feature importance to disk
                    featureImportanceFileName = classifierName + "-featureImportance-split-" + str(splitIndex) + "." + dataPreprocessing + ".csv"
                    with open( os.path.join(folder_name, featureImportanceFileName), "w") as fp :
                        fp.write("feature,importance\n")
                        for featureImportance, featureIndex in featuresByImportance :
                            fp.write( "\"" + variablesX[int(featureIndex)] + "\"," + str(featureImportance) + "\n")
					
                    # also create and plot confusion matrix for test
                    confusionMatrixFileName = classifierName + "-confusion-matrix-split-" + str(splitIndex) + "-" + dataPreprocessing + ".png"
                    confusionMatrix = confusion_matrix(y_test, y_test_pred)
                    plot_confusion_matrix(confusionMatrix, classes, os.path.join(folder_name, confusionMatrixFileName)) 

                except Exception as e :
                    logging.warning("\tunexpected error: ", e)
				
                splitIndex += 1
	    
            # the classifier might have crashed, so we need a check here
            if len(performances[classifierName][dataPreprocessing]) > 0 :
                testPerformance = [ x[0] for x in performances[classifierName][dataPreprocessing] ]
                test_performance_dict = { metric_name : [p[metric_name] for p in testPerformance] for metric_name, metric_performance in testPerformance[0].items() }

                for metric_name in test_performance_dict :
                    logging.info("Average %s (test) of classifier %s on %s data: %.4f (+/- %.4f)" % (metric_name, classifierName, dataPreprocessing, np.mean(test_performance_dict[metric_name]), np.std(test_performance_dict[metric_name])))
                            
                # plot a last confusion matrix including information for all the splits
                confusionMatrixFileName = classifierName + "-confusion-matrix-" + dataPreprocessing + ".png"
                confusionMatrix = confusion_matrix(all_y_test, all_y_pred)
                plot_confusion_matrix(confusionMatrix, classes, os.path.join(folder_name, confusionMatrixFileName)) 

                # but also save all test predictions, so that other metrics could be computed on top of them
                df = pd.DataFrame()
                df["y_true"] = all_y_test
                df["y_pred"] = all_y_pred
                df.to_csv(os.path.join(folder_name, classifierName + "-test-predictions-" + dataPreprocessing + ".csv"), index=False)

    # now, here we can write a final report
    # first, convert performance dictionary to list
    performances_list = []
    for classifier_name in performances :
        for data_preprocessing in performances[classifier_name] :
            
            if len(performances[classifier_name][data_preprocessing]) > 0 :
                performance = [ x[0] for x in performances[classifier_name][data_preprocessing] ]
                performance_mean = np.mean(performance)
                performance_std = np.std(performance)

                performances_list.append( [classifier_name + " (" + data_preprocessing + ")", performance_mean, performance_std, performance] )

    performances_list = sorted(performances_list, key = lambda x : x[1], reverse=True)

    final_report_file_name = os.path.join(folder_name, final_report_file_name)
    logging.info("Final results (that will also be written to file \"" + final_report_file_name + "\"...")

    with open(final_report_file_name, "w") as fp :

        fp.write("Final accuracy results for variable \"%s\", %d samples, %d classes:\n" % (variableY, len(X), len(classes))) 
        
        for result in performances_list :

            temp_string = "Classifier \"%s\", accuracy: mean=%.4f, stdev=%.4f" % (result[0], result[1], result[2])
            logging.info(temp_string)
            fp.write(temp_string + "\n")

            temp_string = "Folds: %s" % str(result[3])
            logging.info(temp_string)
            fp.write(temp_string + "\n\n")


    #		# this part can be skipped because it's computationally expensive; also skip if there are only two classes
    #		if False :
    #			# multiclass classifiers are treated differently
    #			logging.info("Now training OneVsOneClassifier with " + classifierName + "...")
    #			multiClassClassifier = OneVsOneClassifier( classifierData[0] ) 
    #			multiClassClassifier.fit(trainData, trainLabels)
    #			trainScore = multiClassClassifier.score(trainData, trainLabels)
    #			testScore = multiClassClassifier.score(testData, testLabels)
    #			logging.info("\ttraining score: %.4f ; test score: %.4f", trainScore, testScore)
    #			logging.info(common.classByClassTest(multiClassClassifier, testData, testLabels))
    #
    #			logging.info("Now training OneVsRestClassifier with " + classifierName + "...")
    #			currentClassifier = copy.deepcopy( classifierData[0] )
    #			multiClassClassifier = OneVsRestClassifier( currentClassifier ) 
    #			multiClassClassifier.fit(trainData, trainLabels)
    #			trainScore = multiClassClassifier.score(trainData, trainLabels)
    #			testScore = multiClassClassifier.score(testData, testLabels)
    #			logging.info("\ttraining score: %.4f ; test score: %.4f", trainScore, testScore)
    #			logging.info(common.classByClassTest(multiClassClassifier, testData, testLabels))
    #
    #			logging.info("Now training OutputCodeClassifier with " + classifierName + "...")
    #			multiClassClassifier = OutputCodeClassifier( classifierData[0] ) 
    #			multiClassClassifier.fit(trainData, trainLabels)
    #			trainScore = multiClassClassifier.score(trainData, trainLabels)
    #			testScore = multiClassClassifier.score(testData, testLabels)
    #			logging.info("\ttraining score: %.4f ; test score: %.4f", trainScore, testScore)
    #			logging.info(common.classByClassTest(multiClassClassifier, testData, testLabels))
	
	# TODO save files for each classifier:
	#	- recall?
	#	- accuracy?
	#	- "special" stuff for each classifier, for example the PDF tree for DecisionTree
	
    return

if __name__ == "__main__" :
    sys.exit(main())
