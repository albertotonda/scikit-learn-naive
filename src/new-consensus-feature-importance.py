import argparse
import fnmatch
import logging
import os
import math
import pandas as pd
import sys

import common

def main() :

    # a few hard-coded arguments
    percentage_features = 0.1
    top_features = 5
    cutoff_percentage = 0.99 # if 'auto' is specified, only consider predictors that perform at least this percentage of the best
    min_number_of_predictors = 5

    common.initialize_logging()
    args = parse_command_line()

    logging.info("Reading files in directory \"%s\"..." % args.directory)

    file_names = [ f for f in os.listdir(args.directory) if f.endswith(".csv") ]
    logging.info("Found a total of %d files in the directory" % len(file_names))

    if args.files :
        logging.info("Only files with the following patterns will be considered: %s" % str(args.files))

        selected_file_names = []
        for pattern in args.files :
            #selected_file_names.extend( fnmatch.filter(file_names, pattern) ) 
            files_matching_pattern = [ f for f in file_names if f.startswith(pattern + "-") and f.find("featureImportance") != -1 ]
            selected_file_names.extend(files_matching_pattern)

        file_names = selected_file_names

    elif args.auto :
        logging.info("Trying to automatically select a certain number of classifiers/regressors. Looking for \"00_final_result*csv\" file in directory \"%s\"..." % args.directory)

        final_result_file = [ f for f in os.listdir(args.directory) if f.endswith(".csv") and f.startswith("00_final_result") ]

        if len(final_result_file) > 0 :
            logging.info("File found! Analyzing...")
            df = pd.read_csv(os.path.join(args.directory, final_result_file[0]))

            target_metric = ""
            predictor_column = ""
            if "r2 (mean)" in df.columns :
                target_metric = "r2 (mean)"
                predictor_column = "regressor"
                logging.info("Found \"%s\", this looks like a regression problem!" % target_metric)
            elif "F1 (mean)" in df.columns :
                target_metric = "F1 (mean)"
                predictor_column = "classifier"
                logging.info("Found \"%s\", this looks like a classification problem!" % target_metric)

            # find best performance (in both cases, higher is better)
            best_metric_value = df[target_metric].max()
            cutoff_metric_value = cutoff_percentage * best_metric_value
            logging.info("Best value for %s: %.4f; selecting predictors performing at least %.2f%% of this value..." % 
                    (target_metric, best_metric_value, cutoff_percentage * 100.0 ))

            # select predictors that are withing the cutoff percentage of the best performance
            high_performing_predictors = []
            for index, row in df.iterrows() :
                if row[target_metric] >= cutoff_metric_value :
                    high_performing_predictors.append( row[predictor_column] )

            # check: if we do not have enough predictors, keep adding them
            if len(high_performing_predictors) < min_number_of_predictors :
                logging.info("I could not find at least %d high-performing predictors, so I am going to add others until I get to %d" % (min_number_of_predictors, min_number_of_predictors))
                df = df.sort_values(by=target_metric, ascending=False) # sort by performance, descending
                for index, row in df.iterrows() :
                    if len(high_performing_predictors) < min_number_of_predictors :
                        if row[predictor_column] not in high_performing_predictors :

                            heuristic_stop = False

                            # first heuristic check to stop: are we about to include a Dummy classifier/regressor?
                            # second heuristic check: is the target metric REALLY low? (e.g. 0.0?)
                            if row[predictor_column].startswith("Dummy") :
                                logging.info("Reached a row with a Dummy predictor, \"%s\". Including it would make no sense, stopping here." % (row[predictor_column]))
                                heuristic_stop = True
                            elif row[target_metric] <= 0.0 :
                                logging.info("Reached a low value of %s (%.4f) for predictor \"%s\". Stopping here." % (target_metric, row[target_metric], row[predictor_column]))
                                heuristic_stop = True
                            
                            if heuristic_stop :
                                break

                            # if we did not reach the heuristic stop, we add the predictor to the list
                            high_performing_predictors.append(row[predictor_column])


            lowest_metric_value = df[df[predictor_column]==high_performing_predictors[-1]][target_metric]
            logging.info("The following %d predictors will be considered in the analysis: %s" % (len(high_performing_predictors), str(high_performing_predictors))) 
            logging.info("The last classifier included in the list, \"%s\", has a performance of %.4f, that is %.2f%% of the best." % 
                    (high_performing_predictors[-1], lowest_metric_value, lowest_metric_value/best_metric_value * 100.0))

            # convert the list of high-performing predictors to a list of file names to treat
            for p in high_performing_predictors :
                file_names.extend( [ f for f in file_names if f.startswith(p + "-") and f.find("featureImportance") != -1 ] )


    n_files = len(file_names)
    logging.info("A total of %d files was selected: %s" % (len(file_names), str(file_names)))
    sys.exit(0)

    logging.info("Reading files and merging information...")
    all_rankings = dict()
    for f in file_names :

            df = pd.read_csv( os.path.join(args.directory, f), sep=',')
            all_rankings[f] = df.values



    n_features = len(all_rankings[file_names[0]])
    logging.info("The total number of features is %d" % n_features)
    logging.info("%s" % all_rankings[file_names[0]])

    top_features_percentage = math.ceil(percentage_features * n_features)

    logging.info("Now evaluating features that are in the top %.2f%% (%d) or in the top %d" % (percentage_features * 100, top_features_percentage, top_features))

    features_dictionary = { f: {'top': 0, 'percentage':0} for f in all_rankings[file_names[0]][:,0] }
    for f in all_rankings :

            for j in range(0, max(top_features_percentage, top_features)) :
                
                if j < top_features_percentage : features_dictionary[all_rankings[f][j,0]]['percentage'] += 1 
                if j < top_features : features_dictionary[all_rankings[f][j,0]]['top'] += 1

    # and now, some sorting
    list_features_percentage = sorted( [ [f, features_dictionary[f]['percentage']] for f in features_dictionary ], key = lambda x : x[1], reverse=True )
    list_features_top = sorted( [ [f, features_dictionary[f]['top']] for f in features_dictionary ], key = lambda x : x[1], reverse=True )

    logging.info("Features that appear most frequently among the top %.2f%%: %s" % (percentage_features * 100, str(list_features_percentage)))
    for entry in list_features_percentage :
        logging.info("%s (%d, %.2f%%)" % (entry[0], entry[1], float(entry[1])/n_files * 100))

    logging.info("Features that appear most frequently among the top %d: %s" % (top_features, str(list_features_top)))
    for entry in list_features_top :
        logging.info("%s (%d, %.2f%%)" % (entry[0], entry[1], float(entry[1])/n_files * 100))

    return

def parse_command_line() :
	
    parser = argparse.ArgumentParser(description="Script to aggregate feature importance in a single ranking, starting from the results of multiple classification or regression algorithms.\nBy Alberto Tonda, 2019-2022 <alberto.tonda@gmail.com>")
    
    # required argument
    parser.add_argument("-d", "--directory", help="Directory containing all CSV files with relative feature importance.", required=True)	
    
    # list of elements
    parser.add_argument("-f", "--files", nargs='+', help="List of file names. Only files starting with these names will be considered.")

    # optional argument
    parser.add_argument("-t", "--top", type=int, help="Number of top features to consider.")

    # optional argument, output file
    parser.add_argument("-o", "--output", help="Output file")

    # flag, true/false: this will try to automatically select a certain number of classifiers/regressors, depending on performance
    parser.add_argument("-a", "--auto", action='store_true', help="If set, the algorithm will look for a '00_final_result*.csv' file, and use its content to automatically select a restricted number of best-performing predictors, to use as ensemble to establish consensus on feature importance. Heavily heuristic.")
    
    # flag, it's just true/false
    #parser.add_argument("-sz", "--startFromZero", action='store_true', help="If this flag is set, the algorithm will include an individual with genome '0' [0,0,...,0] in the initial population. Useful to improve speed of convergence, might cause premature convergence.")
            
    args = parser.parse_args()
    
    return args

if __name__ == "__main__" :
    sys.exit( main() )
