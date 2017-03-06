import logging
import data_loader as dl
import fast_fuzzy_classifier as ffc
import chi_fuzzy_classifier as cfc
import feature_selection_ga as fsga

import random
from multiprocessing import Pool


logger = logging.getLogger()

def set_logger():
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("\033[92m%(asctime)s %(levelname)s\033[0m\t%(message)s")
    ch.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(ch)

def poker_data_properties():
    file_name = "../data/poker-hand-testing.data"
    cols = range(10)
    class_col = 10

    row_n = int( 10 ** 6 )
    data_n = int( 10 ** 4 )
    def trans_f(x):
        if x[1] != '1':
            x[1] = '2'
        return x

    data_properties = dl.DataProperties(file_name, cols, class_col, row_n, data_n, transformation_fun=trans_f)

    return data_properties

def kddcup_data_properties():
    file_name = "../data/kddcupfull.data"
    cols = [0, 4, 5, 9, 10, 12, 13, 14 , 15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 27, 28, 29,
        30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
    class_col = 41
    row_cnt = 1000
    data_n = 1000

    data_properties = dl.DataProperties(file_name, cols, class_col, row_cnt, data_n)

    return data_properties

def covtype_data_properties():
    file_name = "../data/covtype.data"
    cols = range(54)
    class_col = 54

    row_cnt = 580 * 10 ** 3
    data_n = 10 ** 3
    def filter_f(x):
        return x[1] == "1" or x[1] == "2"

    data_properties = dl.DataProperties(file_name, cols, class_col, row_cnt, data_n, filter_fun=filter_f)

    return data_properties

def random_features(feature_n, sample_n):
    rand_features = []
    
    features = list( range(feature_n) )
    for i in range(sample_n):
        rand_i = int( random.random() * len(features) )
        rand_features.append( features[rand_i] )

    return rand_features

def specific_features(d, features):
    return tuple( [ d[i] for i in features ] )

def extract_features(data, ranges, features):
    extracted_data = [ ( specific_features(d[0], features), d[1] ) for d in data ]
    extracted_ranges = [ ranges[i] for i in features ]

    return extracted_data, extracted_ranges

def random_classifier_and_features(training_data, ranges, feature_n):
    features = random_features(len(training_data[0][0]), feature_n)
    data_sample, ranges_sample = extract_features(training_data, ranges, features)
    clf = ffc.FastFuzzyClassifier(data_sample, ranges_sample)
    clf.fit()

    return clf, features

def get_accuracy(t, classifiers):
    avg = { "1": 0, "2" : 0 }
    for clf in classifiers:
        prediction = clf[0].predict( specific_features(t[0], clf[1]) )
        avg["1"] += prediction["1"]
        avg["2"] += prediction["2"]

    winner = max(avg)
    print(avg[winner])
    if avg[winner] != 0.0 and winner == t[1]:
        return 1
    else:
        return 0

def main():
    verification_data_perc = 0.1

    data_properties = covtype_data_properties()
    data_loader_instance = dl.DataLoader(data_properties)
    data_loader_instance.set_logger(logger)
    data_loader_instance.load(shuffle=True)

    data = data_loader_instance.get_data()
    ranges = data_loader_instance.get_ranges()
    
    verification_data_n = int(verification_data_perc * len(data))
    training_data = data[:-verification_data_n]
    verification_data = data[-verification_data_n:]

    for classifier_n in [ int(2 ** i) for i in [0, 1, 2] ]:
        logger.debug("Using " + str(classifier_n) + " classifiers")
        classifiers = []
        feature_n = int( 10 )
        with Pool(processes=4) as pool:
            parameters =  [ (training_data, ranges, feature_n) for i in range(classifier_n) ]
            classifiers = pool.starmap( random_classifier_and_features, parameters)

        logger.debug("\tClassifying...")

        with Pool(processes=4) as pool:
            acc_list = pool.starmap( get_accuracy, [ (c, classifiers) for c in verification_data ] )

        acc = len([a for a in acc_list if a == 1]) / len(verification_data)
        logger.debug( "\tAccuracy " + str(acc) )

    logger.debug("Class distribution")
    for c in set( [ x[1] for x in verification_data ] ):
        in_class_n = len( [x for x in verification_data if x[1] == c] )
        logger.debug(
            "\tData items with class " + c + " -> " + str( in_class_n ) + \
            "\t percentage " + str( in_class_n / len(verification_data) )
        )

    # logger.debug("Classifying with full data...")
    # clf = ffc.FastFuzzyClassifier(training_data, ranges)
    # clf.set_logger(logger)
    # clf.fit()
    # logger.debug(clf.evaluate(verification_data))

if __name__ == "__main__":
    set_logger()
    main()
