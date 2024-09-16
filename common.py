# Utility scripts to load datasets
# TODO: put logging.info everywhere instead of print
import logging
import numpy as np
import os
import sys

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

def load_regression_data_Guillaume(configuration=3) :
    """
    Data from Guillaume Delaplace and his Ukrainian colleague.
    """
    file_data = "local_files/Agitation_mécanique_3 albertoV2.xlsx"
    df = read_excel(file_data, sheet_name="data-clean")
    
    df.columns = [c if c != "HF/HL exp" else "HFHLexp" for c in df.columns ]
    print(df.columns)

    columns_not_to_be_used = ["N° exp", "type"]
    
    variable_y = "HFHLexp"
    variables_X = []
    
    # depending on the type of experiment, some of the features may or may not
    # be used in X
    if configuration == 0 :
        variables_X = ["Rea", "Ca", "Ntm", "CBsurHL"]
    elif configuration == 1 :
        variables_X = ["Rea", "Ca*", "Ntm", "CBsurHL"]
    elif configuration == 2 :
        variables_X = ["Rea", "Caost*", "Ntm", "CBsurHL", "n'"]
    elif configuration == 3 :
        variables_X = ["Rea", "Ca*rosen", "CBsurHL", "n", "gamma", "Ntm", "N.t*"]
    else :
        variables_X = [c for c in df.columns if c != variable_y and c not in columns_not_to_be_used]
    
    X = df[variables_X].values
    y = df[variable_y].values.reshape(-1, 1)
    
    return X, y, variables_X, [variable_y]
    

def load_regression_data_Deniz() :
    """
    Data obtained from the work of Deniz and Guillaume Delaplace.
    """
    file_data = "local_files/Deniz WPI prediction .XLSX"
    df = read_excel(file_data, header=[0])
    print(df)
    
    # now, the columns here are tuples (...), let's change them to strings;
    # and then, sanitize the result, to avoid weird characters
    df.columns = [slugify(c) for c in df.columns]
    
    # the target column is the only one that contains 'target' in the name
    variable_y = [c for c in df.columns if c.find("3") != -1][0]    
    variables_X = [c for c in df.columns if c != variable_y]
    
    X = df[variables_X].values
    y = df[variable_y].values.reshape(-1,1)
    
    return X, y, variables_X, [variable_y]

def load_regression_data_hybrid_models() :
    """Data set with data used for hybrid models"""
    file_data = "local_files/DatArticle_SVM4.xls"
    df = read_excel(file_data)
    print(df)
    
    # remove all columns that end with '.1' (duplicates)
    columns = [c for c in df.columns if not c.endswith(".1")]
    
    variable_y = "qp"
    variables_X = [c for c in columns if c != variable_y]
    
    X = df[variables_X].values
    y = df[variable_y].values.reshape(-1,1)
    
    return X, y, variables_X, [variable_y]

def loadBeerDataset() :
    """Dataset with 128 different types of beer"""
    
    file_data = "local_files/F0805.xlsx"
    df = read_excel(file_data, sheet_name="Data")
    print(df)
    
    class_column = "Style"
    variablesX = [ c for c in df.columns if c != class_column ]
    
    X = df[variablesX].values
    y = df[class_column].values
    
    return X, y, variablesX, [class_column]

def loadMl4MicrobiomeCRC() :
    """Dataset CRC from the ml4microbiome WG3"""

    file_data = "../ml4microbiome-crc/data/preprocessed-df.csv"
    print("Loading file \"%s\"..." % file_data)
    df = read_csv(file_data)

    # select all columns that start with "msp"
    variablesX = [c for c in df.columns if c.startswith("msp")]
    for v in ["age", "instrument_model", "westernised", "country", "gender"] : variablesX.append(v)
    variablesY = ["health_status"]

    X = df[variablesX].values
    y = df[variablesY].values
    
    return X, y, variablesX, variablesY

def loadMl4Microbiome() :
    """Dataset 'id10' from the ml4microbiome WG3"""

    # hard-coded files
    folder = "../ml4microbiome"
    file_metadata = os.path.join(folder, "HGMA.web.metadata.csv")
    file_data = os.path.join(folder, "HGMA.web.MSP.abundance.matrix.csv")
    dataset_id = "id10"

    # load data as pandas dataframes
    print("Loading file \"%s\"..." % file_metadata)
    df_metadata = read_csv(file_metadata)
    print("Loading file \"%s\"..." % file_data)
    df_data = read_csv(file_data)

    # prepare X (data), y (labels) arrays for classification
    df_dataset = df_metadata[ df_metadata["dataset.ID"] == dataset_id ]
    # get the list of patient samples identifiers
    sample_ids = df_dataset["sample.ID"].values
    # take note of the names of the rows, that will be later the names of the columns of the transposed matrix
    column_names = df_data.iloc[:,0].values
    # now, each sample is a column in df_data, so we need to first select the columns and then turn (transpose) the matrix to have them as rows
    df_data_dataset = df_data[sample_ids].transpose()
    df_data_dataset.columns = column_names

    print(df_data_dataset)

    # feature values are just the data selected so far
    X = df_data_dataset.values
    # labels are 'Case'/'Control' taken from the corresponding metadata column
    y = df_dataset["type"].values

    print("Features:", X)
    print("Labels:", y)

    # for stupid reasons, my old script needs integer labels...
    labels_to_integers = {"Case": 1, "Control": 0}
    y_ = np.zeros(y.shape)
    for i in range(0, y_.shape[0]) :
        y_[i] = labels_to_integers[y[i]]

    return X, y_, column_names, ["Label"]

def loadEasyBenchmark() :
    """Easy benchmark, it's just a dumb function."""
    X = np.arange(0, 4, 0.01)
    y = np.array([ np.sin(2*x)**2 - np.sin(x-1) for x in X ])
    
    X_ = X.reshape((X.shape[0], 1))
    y_ = y.reshape((y.shape[0], 1))

    variablesX = ["x"]
    variablesY = ["y"]

    return X_, y_, variablesX, variablesY


def loadChristianQuestionnaireRegression() :
    data_file = "../questionnaire-rallou-christian/processed_data/FUSED_DATA_Phase3.xlsx"
    df = read_excel(data_file)
    
    irrelevant_columns = ["StartDate", "EndDate", "Progress", "Duration__in_seconds_",
                      "Finished", "RecordedDate", "UserLanguage", "Q1.2"]
    irrelevant_columns.append("age")
    irrelevant_columns.append("Q2.9")
    df = df[[x for x in list(df) if x not in irrelevant_columns]]
    
    # remove all questions that mention 'TEXT' (meaning, people can write something in free form)
    df = df[[x for x in list(df) if 'TEXT' not in x]]
    print("After removing irrelevant and 'TEXT' columns, %d columns are left." % df.shape[1])
    all_columns = list(df)
    
    # we have a list of questions with multiple possible answers
    # that can be checked (all, none, some); empty values can be replaced with '0'
    # but ONLY for those specific questions, elsewhere it's just "answer not given"
    multiple_choice_questions = ["Q2.7", "Q2.8", "Q3.3", "Q4.1", "Q4.4", "Q4.8", "Q4.13"]
    print("Replacing empty answers with '0' in columns: %s" % multiple_choice_questions)
    
    # now, all other missing values will be replaced with -1
    print("Replacing all other missing values with '-1'")
    df.replace(to_replace=' ', value='-1', inplace=True)
    df.replace(to_replace=np.nan, value='-1', inplace=True)
    
    # let's count all rows where there is at least a ' ' left; but it's a little bit more
    # complicated, as question 2.8 can be empty if people replied '1' to question 2.7
    rows_not_answer = 0
    for index, row in df.iterrows() :
        #if ' ' in row[[x for x in all_columns if not x.startswith("Q2.8")]].values : rows_not_answer += 1
        if ' ' in row.values or np.nan in row.values : rows_not_answer += 1
        #for c in all_columns :
        #    if row[c] == ' ' : print("Sample #%d, found missing answer for question \"%s\"" % (index, c))

    # for all columns that start with that question, replace all ' ' with '0'
    replacement_dictionary = dict()
    for mcq in multiple_choice_questions :
        questions = [x for x in list(df) if x.startswith(mcq)]
        for q in questions :
            replacement_dictionary[q] = ' '
    df.replace(to_replace=replacement_dictionary, value='0', inplace=True)
    
    features = list(df)
    variablesX = [f for f in features if not f.startswith("Q26")]
    variableY = "Q26.1_1"
    X = df[variablesX].values
    y = df[variableY].values
    y_ = y.reshape( y.shape[0], 1 ) 
    
    return X, y_, variablesX, [variableY]
    

def loadXORData() :

    data_file = "../xor-experiments/dataset-xor-classification-42.csv"
    df = read_csv(data_file)

    variableY = "class"
    variablesX = [ x for x in list(df) if x != variableY ]

    X = df[variablesX].values
    y = df[variableY].values.reshape(-1)

    return X, y, variablesX, [variableY]

def loadCoronaData() :

    data_file = "../alejandro-corona-cnn/data/data.csv"
    labels_file = "../alejandro-corona-cnn/data/labels.csv"

    df_data = read_csv(data_file)
    df_labels = read_csv(labels_file)

    variableY = "class"
    variablesX = [ "feature-%d" % i for i in range(0, len(list(df_data))) ]

    X = df_data.values
    y = df_labels.values.reshape(-1)

    logging.info("Read data of size %d x %d" % (X.shape[0], X.shape[1]))
    logging.info("Read labels of size %d" % y.shape[0])

    return X, y, variablesX, [variableY]

def loadAndreaMaioranoData() :
    dataset_file = "../agriculture-forecasting-ec-jrc/data/20191119-ML.EU.Wheat.raw.csv"
    logging.info("Reading dataset \"%s\"..." % dataset_file)
    df = read_csv(dataset_file)

    logging.info("The dataset has %d samples and %f features" % (df.shape[0], df.shape[1]))
    
    sys.exit(0)

    return X, y_, variablesX, [variableY_1, variableY_2]


def loadRallouData() :
	
    dataset_file = "../clustering-rallou/data/2020-03-04_Questionnaire_clean_reformattage.csv"
    logging.info("Reading dataset \"%s\"..." % dataset_file)
    df = read_csv(dataset_file, sep=";")

    feature_date = "Horodateur"
    feature_id = "ID"
    irrelevant_questions = ["Q2 ", "Q20", "Q28", "Q29", "Q30", "Q31", "Q32", "Q33"]

    variablesX = [f for f in list(df) if f[0:3] not in irrelevant_questions and f != feature_date and f != feature_id]
    print("The following questions will be considered as features for classification:", variablesX)
    features_not_considered = [f for f in list(df) if f[0:3] in irrelevant_questions]
    print("The following questions will be ignored:", features_not_considered)
	
    X = df[variablesX].values

    variableY = "Class"
    y = np.zeros(X.shape[0])
    for index, row in df.iterrows() :
        if row[features_not_considered[0]] != row[features_not_considered[1]] :
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

import unicodedata
import re

def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')

if __name__ == "__main__" :
    """
    This 'main' function is just here for trials
    """
    
    X, y, variables_X, variables_y = load_regression_data_Guillaume()
    print(X)
    print(y)
    print(variables_X)
    print(variables_y)