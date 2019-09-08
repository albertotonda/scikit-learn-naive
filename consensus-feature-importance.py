
import argparse
import datetime
import logging
import os
import pandas as pd
import re as regex
import sys

from logging.handlers import RotatingFileHandler

def main() :
	# hard-coded values
	summary_file = "00_summary.txt"
	
	# get command-line arguments
	args = parse_command_line()

	# create folder with unique name
	folderName = None
	#folderName = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")  
	#folderName += "-unique-name"
	#if not os.path.exists(folderName) : os.makedirs(folderName)

	# initialize logging, using a logger that smartly manages disk occupation
	initialize_logging(folderName)

	# start program, list all feature importance files and find file with the summary
	logging.info("Reading files in folder \"%s\"..." % args.folder)
	all_files = os.listdir(args.folder)
	feature_importance_files = [ os.path.join(args.folder, f) for f in all_files if 'featureImportance' in f ] 
	
	logging.info("Found a total of %d feature importance files." % len(feature_importance_files))
	if summary_file in all_files :

		logging.info("Summary file found.")
		summary_file = os.path.join(args.folder, summary_file)
		
		# inside the summary file, we have the relative order of goodness of the algorithms
		# so, we need to extract the information from the file
		algorithm_sorted_list = extract_algorithm_order(summary_file)
		
		for a, s in algorithm_sorted_list :
			logging.info("Algorithm %s has R2=%.4f" % (a, s))
	
		# sort the list of files based on the sorted list of algorithms TODO implement cutoff threshold? 
		new_algorithm_sorted_list = []
		new_feature_importance_files = []
		for a, s in algorithm_sorted_list :
			
			# TODO THIS IS WRONG, as "RandomForest" is inside a string like "RandomForest_300"
			
			logging.info("Now gathering feature importance files for algorithm \"%s\"..." % a)
			algorithm_feature_importance_files = []
			
			for f in feature_importance_files :
				
				match = regex.search("(\w+)-(\w+)-featureImportance", f)
				if match and match.group(1) == a : algorithm_feature_importance_files.append(f)

			logging.info("Gathered a total of %d feature importance files." % len(algorithm_feature_importance_files))
			new_feature_importance_files.extend(algorithm_feature_importance_files)
			
			if len(algorithm_feature_importance_files) > 0 : new_algorithm_sorted_list.append(a)
		
		logging.info("Sorted file list: %s" % str(new_feature_importance_files))
		feature_importance_files = new_feature_importance_files
		algorithm_sorted_list = new_algorithm_sorted_list
		
	else :
		logging.info("Summary file not found.")
		# if we have no information, we just use alphabetical sorting of feature files
	
	# get the name of the first algorithm from the first feature file
	file_index = 0
	current_algorithm = regex.search("(\w+)-(\w+)-featureImportance", feature_importance_files[file_index]).group(1)
	logging.info("Current algorithm: \"%s\"" % current_algorithm)
	
	# TODO get feature importance by algorithm, then try to aggregate everything...?
	feature_dictionary = {}
	feature_dictionary[current_algorithm] = {}
	
	while file_index < len(feature_importance_files) :
		
		# check if we are still considering the same algorithm
		algorithm = regex.search("(\w+)-(\w+)-featureImportance", feature_importance_files[file_index]).group(1)
		
		if algorithm == current_algorithm :
			
			# read file, get statitics, add information
			df = pd.read_csv(feature_importance_files[file_index])
			
			# NOTE 	basic idea here: we add up the rank of each feature; at the end,
			#	the feature with the smallest total sum is the most important
			features_in_order = df['feature']
			for i, f in enumerate(features_in_order) :
				if f not in feature_dictionary[current_algorithm] :
					feature_dictionary[current_algorithm][f] = i
				else :
					feature_dictionary[current_algorithm][f] += i
			
			# go on to the next line
			file_index += 1	
		
		else :
			
			# consolidate stats, print them out
			logging.info("Final stats for current algorithm %s" % current_algorithm)
			feature_list = sorted( [ (f,feature_dictionary[current_algorithm][f]) for f in feature_dictionary[current_algorithm] ], key = lambda x : x[1])
			
			for f, s in feature_list :
				logging.info("- Feature \"%s\" (score %d)" % (f, s))
			
			# change current algorithm
			current_algorithm = algorithm
			feature_dictionary[current_algorithm] = {}
			logging.info("Current algorithm changed to: %s" % current_algorithm)
		
	# consolidate stats, print them out
	logging.info("Final stats for current algorithm %s" % current_algorithm)
	feature_list = sorted( [ (f,feature_dictionary[current_algorithm][f]) for f in feature_dictionary[current_algorithm] ], key = lambda x : x[1])
	
	for f, s in feature_list :
		logging.info("- Feature \"%s\" (score %d)" % (f, s))
	
	# merge them all, maybe plot nice picture?
	for algorithm, score in algorithm_sorted_list :
		# TODO build the algorithm_sorted_list so that it keeps the scores

	return

def extract_algorithm_order(summary_file) :
	
	logging.info("Analyzing summary file...")
	lines = []
	with open(summary_file, "r") as fp : lines = fp.readlines()
	
	# scan through all the lines, looking for a certain pattern, fill a list of tuples (algorithm name, score)
	algorithm_list = []
	for line in lines :
		
		match = regex.search("- (\w+), R\^2=([0-9|\.]+)", line)
		
		if match != None :
			logging.info("Found algorithm \"%s\" with R2=%s" % (match.group(1), match.group(2)))
			algorithm_list.append( [match.group(1), float(match.group(2))] )
	
	# sort (descending) based on performance
	algorithm_list = sorted(algorithm_list, key=lambda x : x[1], reverse=True)
	
	return algorithm_list

def initialize_logging(folderName=None) :
	logger = logging.getLogger()
	logger.setLevel(logging.DEBUG)
	formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(message)s', '%Y-%m-%d %H:%M:%S') 

	# the 'RotatingFileHandler' object implements a log file that is automatically limited in size
	if folderName != None :
		fh = RotatingFileHandler( os.path.join(folderName, "log.log"), mode='a', maxBytes=100*1024*1024, backupCount=2, encoding=None, delay=0 )
		fh.setLevel(logging.DEBUG)
		fh.setFormatter(formatter)
		logger.addHandler(fh)

	ch = logging.StreamHandler()
	ch.setLevel(logging.INFO)
	ch.setFormatter(formatter)
	logger.addHandler(ch)
	
	return

def parse_command_line() :
	
	parser = argparse.ArgumentParser(description="Python script that evolves candidate land uses for Small Agricultural Regions.\nBy Francesco Accatino and Alberto Tonda, 2017-2019 <alberto.tonda@gmail.com>")
	
	# required argument
	parser.add_argument("-f", "--folder", help="Folder containing all CSV files with relative feature importance.", required=True)	
	
	# list of elements, type int
	#parser.add_argument("-rid", "--regionId", type=int, nargs='+', help="List of regional IDs. All SARs belonging to regions with these IDs will be included in the optimization process.")
	
	# flag, it's just true/false
	#parser.add_argument("-sz", "--startFromZero", action='store_true', help="If this flag is set, the algorithm will include an individual with genome '0' [0,0,...,0] in the initial population. Useful to improve speed of convergence, might cause premature convergence.")
		
	args = parser.parse_args()
	
	return args

if __name__ == "__main__" :
	sys.exit( main() )
