# Utility scripts to load datasets
# TODO: put logging.info everywhere instead of print
import logging
import numpy as np
import os

from logging.handlers import RotatingFileHandler
from pandas import read_csv 	# excellent package to manipulate CSV files
from pandas import read_excel 	# the same, but for Excel

def initialize_logging(folderName=None) :
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(message)s', '%Y-%m-%d %H:%M:%S') 

    # the 'RotatingFileHandler' object implements a log file that is automatically limited in size
    if folderName != None :
        fh = RotatingFileHandler( os.path.join(folderName, "00_log.log"), mode='a', maxBytes=100*1024*1024, backupCount=2, encoding=None, delay=0 )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return

def loadRallouData() :
	
    dataset_file = "../clustering-rallou/data/2019-08-22-Questionnaire_clean.csv"
    logging.info("Reading dataset \"%s\"..." % dataset_file)
    df = read_csv(dataset_file)

    feature_date = "Horodateur"
    numerical_questions = ["Q2", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10", "Q11", "Q12", "Q13", "Q14", "Q16", "Q17", "Q18", "Q19", "Q20", "Q25", "Q28", "Q31", "Q32"]
	
    variablesX = [ f for f in list(df) if f.split(" ")[0] in numerical_questions and f.split(" ")[0] != "Q2" and f.split(" ")[0] != "Q20" ]
    X = df[variablesX].values

    variableY = "Class"
    y = np.zeros(X.shape[0])
    for index, row in df.iterrows() :
        if row["Q2 - Quel est votre rÃ©gime alimentaire actuel"] != row["Q20 - Dans l'idÃ©al, quel rÃ©gime alimentaire dÃ©sireriez-vous avoir Ã  l'avenir ?"] :
            y[index] = 1
    y_ = y.reshape( y.shape[0], 1 ) 
    y_ = y_.astype('int64')
	
    return X, y_, variablesX, [variableY]

def loadYongShiDataCalibration2(sheet="AP") :
	
	dataset_file = "../ea-ecosystem-services/data/2019-07-European_data_for_calibration_with_PD_group.xls"
	logging.info("Reading dataset \"%s\"..." % dataset_file)
	df = read_excel(dataset_file, sheet_name=sheet)
	
	variablesX = list(df)
	variablesX.remove("CellCode")	
	
	variableY = ""
	for v in variablesX :
		if v.startswith("Y_") :
			variableY = v
	
	variablesX.remove(variableY)
	
	X = df[variablesX].values
	y = df[variableY].values
	y_ = y.reshape( y.shape[0], 1 ) 
	
	return X, y_, variablesX, [variableY]

def loadLaurentBouvierNewData() :

	dataset_file = "../milk-process-pipes-modelling-symbolic-regression/data/2019-07-12-pi_numbers.csv"
	logging.info("Reading dataset \"%s\"..." % dataset_file)
	df = read_csv(dataset_file)
	
	variableY = "pi1"
	variablesX = list(df)
	logging.info("Variables read: %s" % str(variablesX))
	
	variablesX.remove(variableY) # target
	variablesX.remove('pi8') # contains lots of NaN
	variablesX.remove('pi9') # contains lots of NaN
	logging.info("variablesX=%s" % str(variablesX))
	
	X = df[variablesX].values
	y = df[variableY].values
	y_ = y.reshape( y.shape[0], 1 )
	
	return X, y_, variablesX, [variableY]

def loadYongShiDataCalibration() :
	
	logging.info("Reading dataset...")
	df = read_excel("datasets/data-europe-francesco-yong/data_for_calibration.xls", sheet_name="data_test")
	
	variablesX = list(df)
	variablesX.remove("CellCode")
	variablesX.remove("Y_Avp")
	
	variableY = "Y_Avp"
	
	X = df[variablesX].values
	y = df[variableY].values
	y_ = y.reshape( y.shape[0], 1 ) # reshaping necessary to avoid issues with scikit-learn
	
	return X, y_, variablesX, [variableY]

def loadYongShiData() :
	
	logging.info("Reading datasets...")
	df_1 = read_excel("datasets/data-europe-francesco-yong/final_data.xls", sheet_name="LUTableEU")
	df_2 = read_excel("datasets/data-europe-francesco-yong/final_data.xls", sheet_name="EStableEU")
	df_3 = read_excel("datasets/data-europe-francesco-yong/final_data.xls", sheet_name="ID_BCtableEU")
	
	logging.info("Merging datasets...")
	df = df_1.merge(df_2, on='CellCode')
	df = df.merge(df_3, on='CellCode')
	
	logging.info("Final dataset size: %s" % str(df.shape))
	
	# X = EI, W, R, T, OM, RPP (NI, I, P, H)
	variablesX = [
			"HQI",
			"RP_10",
			"WRI_LU10",
			"Eqphos",
			"OM",
			"CS",
			"Recreation",
			"NOx_ret",
			"SEC",
			"HANPP",
			"GNB",
			"Energy_inputs",
			"LVSTK",
			"TIMREMOV",
			"AGRI_WATER",
			"Temp",
			"Prec",
			"TTL1",
			"TTL2",	
			"TTL3",
			"TTL4",
			"TTL5",
			"TTL9",
			"water_content",
			"F311",
			"F312",
			"F313",
			"F321",
			"F322",
			"F323",
			"F324",
			"F331",
			"F332",
			"F333",
			"F334",
			"F335",
			"For_areas",
			"F211",
			"F212",
			"F213",
			"F221",
			"F222",
			"F223",
			"F231",
			"F241",
			"F242",
			"F243",
			"F244",
			"Ag_areas"
			]

	# Y = FOOD_CROPS + FODDER_CROPS + ENERGY_CROPS + TEXTILE_CR
	variablesY = [
			"FOOD_CROPS",
			"FODDER_CRO",
			"ENERGY_CRO",
			"TEXTILE_CR"
			]
	
	# the output variables is actually a sum of columns in variablesY
	df['Y'] = df.apply( lambda row : sum( row[variablesY] ), axis=1 )
	variableY = "Y"
	
	X = df[variablesX].values
	y = df[variableY].values
	y_ = y.reshape( y.shape[0], 1 )

	# old stuff beyond this line
	#df_BC = read_csv("datasets/data-europe-francesco-yong/BC_final.csv")
	#df_ES = read_csv("datasets/data-europe-francesco-yong/ES_final.csv")
	#df_LU = read_csv("datasets/data-europe-francesco-yong/LU_final.csv")
	#
	#df = df_BC.merge(df_ES, on='CellCode')
	#df = df.merge(df_LU, on='CellCode')
	#
	#variables = sorted( list(df) )
	#print("Variables:", variables)
	#
	## remove variables that are not useful
	#variables.remove("ID_x")
	#variables.remove("ID_y")
	#variables.remove("VarName1")
	#print("Variables, cleaned:", variables)

	return X, y_, variablesX, [variableY]

def loadYingYingData() :
	
	df = read_csv("../milk-process-pipes-modelling-symbolic-regression/data/2019-01-25-dataset-yingying.csv")
	
	variables = list(df)
	variablesX = variables[0:9]
	variableY = variables[9] 
	
	X = df[variablesX].values
	y = df[variableY].values
	y_ = y.reshape( y.shape[0], 1 ) # I need a two-dimensional array
	
	return X, y_, variablesX, [variableY]

def loadCleaningDataGermanSpecific() :
	
	df = read_csv("../cleaning-german-delaplace/data/2019-01-24-complete-dataset.csv")
	
	#df = df[ df["Soil"] == "Egg Yolk" ]
	df = df[ df["Soil"] == "Gelatine" ]
	#df = df[ df["Soil"] == "Starch" ]

	variables = list(df)
	variablesX = variables[2:5]
	variableY = variables[8]
	
	X = df[variablesX].values
	y = df[variableY].values
	y_ = y.reshape( y.shape[0], 1 ) # I need a two-dimensional array
	
	return X, y_, variablesX, [variableY]

def loadCleaningDataGerman() :
	
	df = read_csv("../cleaning-german-delaplace/data/2019-01-24-complete-dataset.csv")
	
	# added: remove all lines containing 'NBR', there is no information for that substrate
	df = df[ ~df.Substrate.str.contains("NBR") ]
	logging.info("A total of %d lines have been selected." % df.shape[0])

	variables = list(df)
	variablesX = variables[2:8]
	variablesX.extend( variables[9:] )
	variableY = variables[8]
	
	X = df[variablesX].values
	y = df[variableY].values
	y_ = y.reshape( y.shape[0], 1 ) # I need a two-dimensional array
	
	#print("X=", X)
	#print("y=", y)
	
	return X, y_, variablesX, [variableY]

def loadAlejandroValidation() :
	
	dfX_train = read_csv("../alejandro-cancer-feature-selection/data/data.matrix", sep='\s+')
	dfy_train = read_csv("../alejandro-cancer-feature-selection/data/labels.txt")
	
	dftest = read_csv("datasets/alejandro-new-data.csv")
	variables = list(dftest)
	variablesX = variables[1:]
	variableY = variables[0]
	
	X_train = dfX_train[variablesX].to_matrix()
	y_train = dfy_train.to_matrix() 
	X_test = dftest[variablesX].to_matrix()
	y_test = dftest[variableY].to_matrix()
	
	return X_train, y_train, X_test, y_test, variablesX, variableY

def loadVougas() :
	
	dfX = read_csv('datasets/vougas_input.csv', delimiter=' ')
	dfY = read_csv('datasets/vougas_output_CH5424802.csv', delimiter=' ')
	
	return dfX.as_matrix(), dfY.as_matrix().ravel()

def loadInsectsClassification() :
	
	fileName = 'datasets/insects-magda-1999-2017_last_version.csv'
	dataframe = read_csv(fileName, delimiter=',').dropna()
	variablesNames = list(dataframe)
	
	X = dataframe.as_matrix()[:,5:]
	y = dataframe.as_matrix()[:,1]

	return X, np.asarray(y, dtype=int), variablesNames[4:], variablesNames[1]

def loadInsects() :
	
	fileName = 'datasets/insects-magda-1999-2017_last_version.csv'
	dataframe = read_csv(fileName, delimiter=',').dropna()
	variablesNames = list(dataframe)
	
	X = dataframe.as_matrix()[:,5:]
	y = dataframe.as_matrix()[:,0]
	
	# I need a 2-dimensional array
	y_ = y.reshape( y.shape[0], 1 )
	
	return X, y_, variablesNames[5:], [variablesNames[0]]

def loadMilkProcessPipes() :

	#fileName = 'datasets/milk-process-pipes.csv'
	fileName = '../milk-process-pipes-modelling-symbolic-regression/utils/old_all_data.csv'
	dataset = read_csv(fileName)
	header = list(dataset)
	
	X = dataset.as_matrix()[:,2:]
	y = dataset.as_matrix()[:,1]
	
	# I need a 2-dimensional array
	y_ = y.reshape( y.shape[0], 1 ) 
	
	return X, y_, header[2:], [header[1]] 

def loadMilkProcessPipesDimensionalAnalysis() :
	
	fileName = '../milk-process-pipes-modelling-symbolic-regression/data/2018-10-18-dimensional-analysis-prepared-for-ml.csv'
	dataset = read_csv(fileName)
	header = list(dataset)
	
	# two equations, F_8 and F_10
	#C = F_8(D, Fr, A)
	index_y = 3
	index_x = [1, 4, 6]
	#ftm = F_10(D, Re, A)
	#index_y = 7
	#index_x = [0, 1, 6]
	
	X = dataset.as_matrix()[:,index_x]
	y = dataset.as_matrix()[:,index_y]
	
	# need a 2-dimensional array
	y_ = y.reshape( y.shape[0], 1 )
	
	return X, y_, [ header[i] for i in index_x ], [header[index_y]]
	

def loadEcosystemServices() :
	
	fileName = 'datasets/dataSARcomplete2006.csv'
	dataset = read_csv(fileName)
	header = list(dataset)
	
	X = dataset.as_matrix()[:,4:]
	y = dataset.as_matrix()[:,:4] # the 'objectives' are in the first 3 columns
	
	return X, y, header[4:], header[:4]

def loadAlejandroNewDataset() :
	
	fileNameX = "../alejandro-cancer-feature-selection/data/data.matrix"
	fileNameY = "../alejandro-cancer-feature-selection/data/labels.txt"

	X = read_csv(fileNameX, sep='\s+').as_matrix()
	y = read_csv(fileNameY).as_matrix().ravel()
	
	biomarkers = []
	with open("../alejandro-cancer-feature-selection/data/idmir.vector") as fp :
		lines = fp.readlines()
		lines.pop(0)
		biomarkers = [ l.rstrip() for l in lines ]
	
	return X, y, biomarkers, "class"

def loadMarcoSoil() :
	
	fileName = "datasets/marco-soil.csv"
	dataframe = read_csv(fileName)
	
	#variableY = '%WEOC_C_mean'
	variableY = 'WEOC_mean'
	variablesX = [ 
				'freq_prairie',
				'diversit?_cult',
				'Beven index',	
				'clay',
				'silt',	
				'sand',	
				'pH_sol_Arras',
				'Cu_edta',
				'Si',
				'Al',
				'Fe'
			]
	
	y = dataframe[variableY].values
	X = dataframe[variablesX].values
	
	print("y=", y)
	print("X=", X)
	
	return X, y, variablesX, variableY


def loadEureqaRegression() :
	
	X = []
	y = []
	variableToIndex = dict()

	targetVariable = "y"
	fileName = "datasets/eureqa-example.csv"

	print("Loading file \"" + fileName + "\"...")
	with open(fileName, "r") as fp :
		lines = fp.readlines()
		variables = lines.pop(0)
		for i, v in enumerate(variables.rstrip().split(',')) : variableToIndex[v] = i 
		for line in lines : 
			tokens = [ float(x) for x in line.rstrip().split(',') ]
			localData = []
			for i, t in enumerate(tokens) :
				if variableToIndex[targetVariable] == i :
					y.append( t )
				else :
					localData.append( t )
			X.append( localData )
	
	return np.array(X), np.array(y)

def loadBiscuitRegression() :

	X_train = []
	y_train = []
	X_test = []
	y_test = []
	
	fileTrain = "datasets/biscuit-example-training.csv"
	fileTest = "datasets/biscuit-example-test.csv"
	
	# order of the columns: t, tf, bf, wl, c, h 

	print("Loading training file \"" + fileTrain + "\"...")
	with open(fileTrain, "r") as fp :
		lines = fp.readlines()
		variables = lines.pop(0)
		
		for line in lines :
			tokens = [ float(x) for x in line.rstrip().split(',') ]
			X_train.append( [ tokens[0], tokens[1], tokens[2] ] )
			y_train.append( [ tokens[3], tokens[4], tokens[5] ] )
		
		
	
	print("Loading training file \"" + fileTrain + "\"...")
	with open(fileTest, "r") as fp :
		lines = fp.readlines()
		variables = lines.pop(0)

		for line in lines :
			tokens = [ float(x) for x in line.rstrip().split(',') ]
			X_test.append( [ tokens[0], tokens[1], tokens[2] ] )
			y_test.append( [ tokens[3], tokens[4], tokens[5] ] )
	
	return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test) 

def loadAromoptiRegression() :
	X_train = []
	X_test = []
	y_train = []
	y_test = []
	
	fileTrain = "datasets/aromopti-training.csv"
	fileTest = "datasets/aromopti-test.csv"
	
	# in this dataset:
	# X_index = 0...5 
	# y_index = 6...14

	print("Loading training file \"" + fileTrain + "\"...") 
	with open(fileTrain, "r") as fp :
	
		lines = fp.readlines()
		variables = lines.pop(0)
		
		for line in lines :
			tokens = [ float(x) for x in line.rstrip().split(',') ]
			X_train.append( [ tokens[0], tokens[1], tokens[2], tokens[3], tokens[4], tokens[5] ] )
			y_train.append( [ tokens[6], tokens[7], tokens[8], tokens[9], tokens[10], tokens[11], tokens[12], tokens[13], tokens[14] ] )

	print("Loading test file \"" + fileTest + "\"...") 
	with open(fileTest, "r") as fp :
	
		lines = fp.readlines()
		variables = lines.pop(0)
		
		for line in lines :
			tokens = [ float(x) for x in line.rstrip().split(',') ]
			X_test.append( [ tokens[0], tokens[1], tokens[2], tokens[3], tokens[4], tokens[5] ] )
			y_test.append( [ tokens[6], tokens[7], tokens[8], tokens[9], tokens[10], tokens[11], tokens[12], tokens[13], tokens[14] ] )
	
	return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

def loadOmineeClassification():
	X_train = []
	X_test = []
	y_train = []
	y_test = []
	
	fileTrain = "datasets/jobTitle-10928-training.csv"
	fileTest = "datasets/jobTitle-10928-test.csv"

	print("Loading training file \"" + fileTrain + "\"...")
	with open(fileTrain, "r") as fp :
		
		lines = fp.readlines()
		variables = lines.pop(0)
		
		for line in lines :
			tokens = [ int(x) for x in line.rstrip().split(',') ]
			y_train.append( tokens.pop(0) )
			X_train.append( tokens )

	print("Loading test file \"" + fileTest + "\"...")
	with open(fileTest, "r") as fp :
		
		lines = fp.readlines()
		variables = lines.pop(0)
		
		for line in lines :
			tokens = [ int(x) for x in line.rstrip().split(',') ]
			y_test.append( tokens.pop(0) )
			X_test.append( tokens )

	return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

# test accuracy of the resulting classifier on all classes
def classByClassTest(classifier, testData, testLabels) :
	
	report = "Class-by-class test of classifier" + str(classifier) + "\n"
	
	classes = sorted(set(testLabels))
	
	# class-by-class performance
	predictedClasses = classifier.predict(testData)
	classPerformance = { cl : {'TPR': 0.0, 'FPR': 0.0, 'correct' : 0.0, 'misplaced' : 0.0, 'noclass' : 0.0} for cl in classes }
	for i in range(0, len(testLabels)) :
		
		if predictedClasses[i] == testLabels[i] :
			classPerformance[testLabels[i]]['correct'] += 1.0
		elif predictedClasses[i] != testLabels[i] and predictedClasses[i] != None :  
			classPerformance[testLabels[i]]['misplaced'] += 1.0
		else :
			classPerformance[testLabels[i]]['noclass'] += 1.0
	
	# now, let's write a report
	correct = sum( classPerformance[x]['correct'] for x in classPerformance ) / float(len(testLabels))
	misplaced = sum( classPerformance[x]['misplaced'] for x in classPerformance ) / float(len(testLabels))
	noclass = sum( classPerformance[x]['noclass'] for x in classPerformance ) / float(len(testLabels))
	
	report += "Totals: correct=%.4f" % correct
	report += ", misplaced=%.4f" % misplaced
	report += ", noclass=%.4f" % noclass
	report += "\n"
	
	# class-by-class printing stuff
	for cl in classes :
		occurrences = float( len([x for x in testLabels if x == cl]) ) 
		report += "Class #" + str(cl) #+ " (" + classesNames[cl] + "):" 
		report +=" correct=%.4f" % float(classPerformance[cl]['correct']/occurrences)
		report +=" [" + str(int(classPerformance[cl]['correct'])) + "/" + str(int(occurrences)) + "]"

		report +=", misplaced=%.4f" % float(classPerformance[cl]['misplaced']/occurrences)
		report +=" [" + str(int(classPerformance[cl]['misplaced'])) + "/" + str(int(occurrences)) + "]"

		report +=", noclass=%.4f" % float(classPerformance[cl]['noclass']/occurrences)
		report +=" [" + str(int(classPerformance[cl]['noclass'])) + "/" + str(int(occurrences)) + "]"
		report +="\n"

	return report
