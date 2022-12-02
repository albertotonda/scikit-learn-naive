import argparse
import fnmatch
import logging
import os
import math
import pandas
import sys

import common

def main() :

    # a few hard-coded arguments
    percentage_features = 0.1
    top_features = 5

    common.initialize_logging()
    args = parse_command_line()

    logging.info("Reading files in directory \"%s\"..." % args.directory)

    file_names = [ f for f in os.listdir(args.directory) if f.endswith(".csv") ]

    if args.files :
        logging.info("Only files with the following patterns will be considered: %s" % str(args.files))

        selected_file_names = []
        for pattern in args.files :
            selected_file_names.extend( fnmatch.filter(file_names, pattern) ) 

        file_names = selected_file_names

    n_files = len(file_names)
    logging.info("A total of %d files was selected: %s" % (len(file_names), str(file_names)))

    logging.info("Reading files and merging information...")
    all_rankings = dict()
    for f in file_names :

            df = pandas.read_csv( os.path.join(args.directory, f), sep=',')
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
    
    # list of elements, type int
    parser.add_argument("-f", "--files", nargs='+', help="List of file names. Only files starting with these names will be considered.")

    # optional argument
    parser.add_argument("-t", "--top", type=int, help="Number of top features to consider.")
    
    # flag, it's just true/false
    #parser.add_argument("-sz", "--startFromZero", action='store_true', help="If this flag is set, the algorithm will include an individual with genome '0' [0,0,...,0] in the initial population. Useful to improve speed of convergence, might cause premature convergence.")
            
    args = parser.parse_args()
    
    return args

if __name__ == "__main__" :
    sys.exit( main() )
