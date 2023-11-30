# Simple Python script that iterates over several regression methods and tests them on a given dataset
# by Alberto Tonda, 2016-2022 <alberto.tonda@gmail.com>

import argparse
import copy
import datetime
import logging
import multiprocessing # this is used just to assess number of available processors
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
import sys

# this is to get the parameters in a function
from inspect import signature, Parameter

# my home-made stuff
import common
#from keraswrappers import ANNRegressor
from polynomialmodels import PolynomialRegressor
#from humanmodels import HumanRegressor

# scikit-learn stuff
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, ShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import all_estimators

# now, this is new and exciting stuff: Generalized Additive Models (GAM) from module pyGAM
#from pygam import LinearGAM 

# but not only, here comes the state of the art for ensemble models, XGBoost and LightGBM
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

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
    # TODO actually, it would be better to go through the ensemble, element by element
    # and then ask those what are the most frequent features
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

    # hard-coded values
    numberOfSplits = 10 # TODO change number of splits from command line
    reference_metric = "r2" # metric that is used by default to sort the regressors
    target_column = 'target'

    # metrics considered
    metrics = dict()
    metrics["r2"] = r2_score
    metrics["explained_variance"] = explained_variance_score
    metrics["MAE"] = mean_absolute_error
    metrics["MSE"] = mean_squared_error
    metrics["MA%E"] = mean_absolute_percentage_error

    # parsing command-line arguments
    # TODO does it make sense to have more than one target variable?
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", help="Set a specific random seed. If not specified, it will be set through system time.", type=int)
    parser.add_argument("--csv", help="Data-set in CSV format. A column must be marked as 'target' and will be used as such. If no 'target' is specified, another column name must be specified through command-line argument '--target'")
    parser.add_argument("--target", help="Name of the target column. It's only used if '--csv' is specified and no 'target' column is found in the data-set.")
    parser.add_argument("--folds", help="Name of the CSV dataset column that will be used to specify the folds. If not specified, data will be randomly split")
    args = parser.parse_args()
    
    # set graphic options
    sns.set_style('darkgrid')

    # let's create a folder with a unique name to store results
    folderName = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + "-regression" 
    if not os.path.exists(folderName) : os.makedirs(folderName)
    
    # initialize logging
    common.initialize_logging(folderName)

    # generate a random seed that will be used for all the experiments (or just use the one given from command line)
    random_seed = int(datetime.datetime.now().timestamp())
    if args.random_seed is not None : random_seed = args.random_seed
    logging.info("Random seed that will be used for all experiments: %d" % random_seed)

    # set the numpy random number generator with the seed
    np.random.seed(random_seed)

    # automatically create the list of regressors
    regressor_dict = dict()
    estimators = all_estimators(type_filter="regressor")

    # NOTE/TODO add scikit-learn-compatible classifiers from other sources
    estimators.append(("XGBRegressor", XGBRegressor))
    estimators.append(("LightGBMRegressor", LGBMRegressor))
    estimators.append(("CatBoostRegressor", CatBoostRegressor))
    # TODO implement a pygamregressor?

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

            # if the classifier has an "n_job" parameter (number of processors to use in parallel, add it
            if 'n_jobs' in params :
                params_dict['n_jobs'] = max(multiprocessing.cpu_count() - 1, 1) # use maximum available minus one; if it is zero, just use one

            # create the name of the regressor, starting from the class string
            regressor_name = name 

            # add the regressor
            regressor_dict[regressor_name] = class_(**params_dict)

            # if it accepts a parameter called 'n_estimators', let's create a second instance and go overboard 
            if 'n_estimators' in params :
                params_dict['n_estimators'] = 300
                regressor_dict[regressor_name + "_300"] = class_(**params_dict)

            # if it's the DummyRegressor, let's add a second version that always predicts the median (default always predicts the mean)
            if class_.__name__ == "DummyRegressor" :
                params_dict['strategy'] = 'median'
                regressor_dict[regressor_name + "_median"] = class_(**params_dict)

        except Exception as e :
            logging.error("Cannot instantiate regressor \"%s\" (exception: \"%s\"), skipping..." % (name, str(e))) 

    # add other regressors, taken from local libraries
    regressor_dict["PolynomialRegressor_2"] = PolynomialRegressor(2)
    regressor_dict["PolynomialRegressor_3"] = PolynomialRegressor(3)

    ### THIS PART IS JUST USED FOR DEBUGGING, JUST BRUTALLY SELECTING REGRESSORS TO HAVE FASTER TESTS
    #regressor_dict = { regressor_name : regressor for regressor_name, regressor in regressor_dict.items() if regressor_name.startswith("RandomForest") }
    ### END OF THE DEBUGGING PART

    logging.info("A total of %d regressors will be used: %s" % (len(regressor_dict), str(regressor_dict)))

    # setting up variables
    X = y = X_train = X_test = y_train = y_test = variablesX = variablesY = None

    if args.csv is not None :
        df = pd.read_csv(args.csv)
        
        # check if the target column exists, or if it has been specified on command line
        if target_column not in df.columns :
            if args.target is not None :
                target_column = args.target
            else :
                logging.error("Cannot find column \"%s\" in CSV file \"%s\". You'll need to specify a different target column from command line. Aborting..." % (target_column, args.csv))
                sys.exit(0)

        # before converting the dataframe to just numpy arrays, let's try to identify the categorical variables
        # and replace them with numerical values, to avoid issues with regressors that cannot deal with them
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist() 
        if len(categorical_columns) > 0 :
            logging.info("Found column(s) with categorical variables \"%s\", converting them to integers..." % str(categorical_columns))

        # replace each categorical column with values (in a naive way, just ascending numbers in the order they were found)
        for cc in categorical_columns :
            # get list of values
            values = df[cc].unique()
            indexes = range(len(values))

            # replace values with their index
            df.replace(to_replace=values, value=indexes, inplace=True)

        # if everything is in place, parse the dataframe
        variablesY = [target_column]
        variablesX = [ c for c in df.columns if c not in variablesY ]

        # get numpy arrays
        X = df[variablesX].values
        y = df[variablesY].values
    
    else :
        # this is just a dumb benchmark
        logging.info("No other dataset was specified on the command line, so a standard (easy) benchmark will be used instead...")
        X, y, variablesX, variablesY = common.loadEasyBenchmark()
    
    logging.info("Regressing %d output variables, in function of %d input variables..." % (y.shape[1], X.shape[1]))
    
    # if the names of the variables are not specified, let's specify them!
    if variablesY is None : variablesY = [ "y" + str(i) for i in range(0, len(y[0])) ]
    if variablesX is None : variablesX = [ "X" + str(i) for i in range(0, len(X[0])) ]

    performances = dict()
    all_folds_indexes = None

    for variableIndex, variableY in enumerate(variablesY) :

        logging.info("** Now evaluating models for variable \"%s\"... **" % variableY)

        # obtain data
        y_ = y[:,variableIndex].ravel()

        # assume here that you will have train/test indexes instead
        # it's also easier for the plots, as we do not face the issue
        # of duplicate values (e.g. same value with two indexes)
        rs = ShuffleSplit(n_splits=numberOfSplits, random_state=random_seed)
        #rs = LeaveOneOut()

        # initialize performance dictionary of arrays
        performances[variableY] = dict()
        for regressorName, regressor in regressor_dict.items() : 
            performances[variableY][regressorName] = dict()
            for metric_name, metric_function in metrics.items() :
                performances[variableY][regressorName][metric_name] = []
            performances[variableY][regressorName]["predicted"] = []
            
        # this is used to store all values of each fold, in order; maybe there's a smarter way to do it
        foldPointsInOrder = []
        # this is used to store the index of the fold in which a point appears in the test set
        fold_point_test_indexes = np.zeros((y_.shape[0],))
            
        # and now, for every regressor
        all_folds_indexes = enumerate(rs.split(X))
        for foldIndex, indexes in all_folds_indexes :

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
            
            # this 'reshape' is unfortunately needed by StandardScaler
            y_train = scalerY.fit_transform(y_train.reshape(-1,1))
            y_test = scalerY.transform(y_test.reshape(-1,1))
            
            # now, we store points of the folder in order of how they appear
            foldPointsInOrder.extend( list(scalerY.inverse_transform(y_test)) )
            
            # also keep track of indexes of folds when points appear in test
            fold_point_test_indexes[test_index] = foldIndex
            
            for regressorIndex, regressorData in enumerate(regressor_dict.items()) :

                regressorName, regressor = regressorData
            
                logging.info( "Fold #%d/%d: training regressor #%d/%d \"%s\"" % (foldIndex+1, numberOfSplits, regressorIndex+1, len(regressor_dict), regressorName) )

                try :
                    regressor.fit(X_train, y_train.ravel())
                    
                    # get predictions
                    y_test_predicted = regressor.predict(X_test)
                    y_train_predicted = regressor.predict(X_train) # TODO at the moment, this is not used

                    # for each metric, call the corresponding function and compute the result
                    for metric_name, metric_function in metrics.items() :
                        performances[variableY][regressorName][metric_name].append( metric_function(y_test, y_test_predicted) )
                        logging.info("- %s score (test): %.4f" % (metric_name, performances[variableY][regressorName][metric_name][-1]))

                    # also record the predictions, to be used later in a global figure
                    performances[variableY][regressorName]["predicted"].extend( list(scalerY.inverse_transform(y_test_predicted.reshape(-1,1))) )
                    
                    # plots
                    import matplotlib.pyplot as plt

                    # plotting first figure, with points 'x' and 'o'
                    y_predicted = regressor.predict(scalerX.transform(X)) # 'X' was never wholly rescaled before
                    y_train_predicted = regressor.predict(X_train)
                    
                    # TODO actually use PROPER matplotlib syntax to deal with this
                    plt.figure()
                    
                    plt.scatter(train_index, y_train, c="gray", label="training data")
                    plt.scatter(test_index, y_test, c="green", label="test data")

                    plt.plot(np.arange(len(y_predicted)), y_predicted, 'x', c="red", label="regression")
                    plt.xlabel("order of data samples")
                    plt.ylabel("target")
                    plt.title(regressorName + ", R^2=%.4f (test)" % performances[variableY][regressorName]["r2"][-1])
                    plt.legend()
                    
                    logging.info("Saving figure...")
                    plt.savefig( os.path.join(folderName, regressorName + "-" + variableY + "-fold-" + str(foldIndex+1) + ".png"), dpi=300 )
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

                    plt.savefig( os.path.join(folderName, regressorName + "-" + variableY + "-fold-" + str(foldIndex+1) + "-b.png"), dpi=300 )
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
            
                except Exception as e :
                    logging.error("Regressor \"" + regressorName + "\" failed on variable \"" + variableY + "\":", e)

    logging.info("Final summary:")
            
    for variableY in variablesY :

        logging.info("For variable \"" + variableY + "\"")

        # we will start creating a dictionary that will be later converted to a Pandas DataFrame and sorted
        # we are going to create a different CSV file for each regression variable
        df_dict = dict()
        df_dict["regressor"] = []
        for metric_name in metrics :
            df_dict[metric_name + " (mean)"] = []
            df_dict[metric_name + " (std)"] = []

        # go over the metrics and print out some statistics
        for regressorName, regressorScores in performances[variableY].items() :

            logging.info("For regressor \"%s\":" % regressorName)
            df_dict["regressor"].append(regressorName)

            for metric_name, metric_function in metrics.items() : 
                metric_mean = np.mean(regressorScores[metric_name])
                metric_std = np.std(regressorScores[metric_name])
                logging.info("\t- %s, mean=%.4f (std=%.4f)" % (metric_name, metric_mean, metric_std))

                df_dict[metric_name + " (mean)"].append(metric_mean)
                df_dict[metric_name + " (std)"].append(metric_std)

            # also, plot a "global" graph
            # issue here, if a regressor fails, you have incongruent matrixes: a check is in order
            if len(foldPointsInOrder) == len( regressorScores["predicted"] ) :
                
                fig = plt.figure()
                ax = fig.add_subplot(111)
                
                #bottom_left_corner = [min(foldPointsInOrder), max(foldPointsInOrder)]
                #top_right_corner = [min(regressorScore["predicted"]), max(regressorScore["predicted"])]
                x_bottom_top = [0, max(foldPointsInOrder)]
                y_bottom_top = [0, max(foldPointsInOrder)]
                
                x_bottom_top = [min(foldPointsInOrder)[0], max(foldPointsInOrder)[0]]
                y_bottom_top = [min(regressorScores["predicted"])[0], max(regressorScores["predicted"])[0]]
                
                x_figure = [x[0] for x in foldPointsInOrder]
                y_figure = [x[0] for x in regressorScores["predicted"]]

                sns.lineplot(x=x_bottom_top, y=y_bottom_top, color='k', linestyle='--', label="1:1") # line
                sns.lineplot(x=x_bottom_top, y=[y_bottom_top[0]*1.20, y_bottom_top[1]*1.20], color='r', linestyle='--', label="20% error")
                sns.lineplot(x=x_bottom_top, y=[y_bottom_top[0]*0.80, y_bottom_top[1]*0.80], color='r', linestyle='--')
                sns.scatterplot(x=x_figure, y=y_figure, color='g', marker='o', alpha=0.7) # points
                
                ax.set_title(regressorName + " measured vs predicted, " + variableY + " (all test)")
                ax.set_xlabel("measured")
                ax.set_ylabel("predicted")
                ax.legend(loc='best')
    
                plt.savefig( os.path.join(folderName, regressorName + "-" + variableY + "-global-b.png"), dpi=300 )
                plt.close(fig)
                
                # another thing we can do, which is clearly useful, is to save all predictions vs values
                # extremely important if we need to recompute R2 or other metrics after the run
                dict_all_predictions = {"test_fold" : fold_point_test_indexes,
                                        "y_true" : [x[0] for x in foldPointsInOrder], 
                                        "y_pred" : [x[0] for x in regressorScores["predicted"]]}
                            
                df_all_predictions = pd.DataFrame.from_dict(dict_all_predictions)
                df_all_predictions.to_csv(os.path.join(folderName, regressorName + "-" + variableY + "-all-test-predictions.csv"), index=False)

        # here, we finished the loop; create a dataframe from the dictionary, then sort it by the key variable
        df = pd.DataFrame.from_dict(df_dict)
        df.sort_values(reference_metric + " (mean)", inplace=True, ascending=False)
        df.to_csv(os.path.join(folderName, "00_final_result_" + variableY + ".csv"), index=False)
        
        # TODO: finally, close the logs
        
        return

# stuff to make the script more proper
if __name__ == "__main__" :
    sys.exit(main())
