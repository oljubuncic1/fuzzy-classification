import logging
import fuzzy_classification.util.data_loader as dl
import fuzzy_classification.classifiers.fast_fuzzy_classifier as ffc
import fuzzy_classification.classifiers.chi_fuzzy_classifier as cfc
import fuzzy_classification.feature_selection.feature_selection_ga as fsga
from fuzzy_classification.classifiers.RandomFuzzyForest import RandomFuzzyForest

import random
from multiprocessing import Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

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

    row_n = int( 10 ** 3 )
    data_n = int( 10 ** 3 )
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
    row_cnt = 1 * 10 ** 6
    data_n = 10 ** 4

    def trans_f(x):
        if x[1] != 'normal.':
            x[1] = '1'
        else:
            x[1] = '2'
        return x

    data_properties = dl.DataProperties(file_name, cols, class_col, row_cnt, data_n)

    return data_properties

def covtype_data_properties():
    file_name = "../data/covtype.data"
    cols = range(54)
    class_col = 54

    row_cnt = 1 * 10 ** 2
    data_n = 10 ** 2
    def filter_f(x):
        return x[1] == "1" or x[1] == "2"

    data_properties = dl.DataProperties(file_name, cols, class_col, row_cnt, data_n, filter_fun=filter_f)

    return data_properties

def random_features(total_feature_n, sample_feature_n):
    rand_features = []
    
    features = list( range(total_feature_n) )
    for i in range(sample_feature_n):
        rand_i = int( random.random() * len(features) )
        rand_features.append( features[rand_i] )

    return rand_features

def specific_features(d, features):
    return tuple( [ d[i] for i in features ] )

def extract_features(data, ranges, features):
    extracted_data = [ ( specific_features(d[0], features), d[1] ) for d in data ]
    extracted_ranges = [ ranges[i] for i in features ]

    return extracted_data, extracted_ranges

def random_with_replacement(data, sample_n):
    sample = set()
    for i in range(sample_n):
        ind = int(random.random() * len(data))
        sample.add( data[ind] )

    return [list(s) for s in sample]

def random_classifier_and_features(training_data, ranges, sample_feature_n):
    total_feature_n = len(training_data[0][0])
    features = random_features(total_feature_n, sample_feature_n)

    data_sample_n = int( 1.0 * len(training_data) )
    data_sample = random_with_replacement(training_data, data_sample_n)
    
    data_sample, ranges_sample = extract_features(data_sample, ranges, features)
    clf = ffc.FastFuzzyClassifier(data_sample, ranges_sample, label_cnt=2)
    clf.fit()

    return clf, features

def get_accuracy(t, classifiers):
    avg = { "1": 0, "2" : 0 }
    for clf in classifiers:
        prediction = clf[0].predict( specific_features(t[0], clf[1]) )
        avg["1"] += prediction["1"]
        avg["2"] += prediction["2"]

    winner = max(avg)
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

    x = [list(t[0]) for t in training_data]
    y = [t[1] for t in training_data]

    rff = RandomFuzzyForest()
    rff.fit(x, y)

    # x = [list(t[0]) for t in training_data]
    # y = [t[1] for t in training_data]
    # # clf = RandomForestClassifier(n_estimators=20, n_jobs=-1, verbose=1)
    # clf = KNeighborsClassifier(n_jobs=-1, n_neighbors=30)
    # clf.fit(x, y)

    # v_x = [list(t[0]) for t in verification_data]
    # v_y = [t[1] for t in verification_data]
    # score = clf.score(v_x, v_y)
    # logger.debug( score )

    # return

    # for classifier_n in [ int(2 ** i) for i in [1, 2, 3, 4] ]:
    #     logger.debug("Using " + str(classifier_n) + " classifiers")
    #     classifiers = []
    #     feature_n = int(5)
    #     with Pool(processes=4) as pool:
    #         parameters =  [ (training_data, ranges, feature_n) for i in range(classifier_n) ]
    #         classifiers = pool.starmap( random_classifier_and_features, parameters)

    #     logger.debug("\tClassifying...")

    #     with Pool(processes=4) as pool:
    #         acc_list = pool.starmap( get_accuracy, [ (c, classifiers) for c in verification_data ] )

    #     acc = len([a for a in acc_list if a == 1]) / len(verification_data)
    #     logger.debug( "\tAccuracy " + str(acc) )

    # logger.debug("Class distribution")
    # for c in set( [ x[1] for x in verification_data ] ):
    #     in_class_n = len( [x for x in verification_data if x[1] == c] )
    #     logger.debug(
    #         "\tData items with class " + c + " -> " + str( in_class_n ) + \
    #         "\t percentage " + str( in_class_n / len(verification_data) )
    #     )

    # logger.debug("Classifying with full data...")
    # clf = ffc.FastFuzzyClassifier(training_data, ranges)
    # clf.set_logger(logger)
    # clf.fit()
    # logger.debug(clf.evaluate(verification_data))

if __name__ == "__main__":
    set_logger()
    main()
