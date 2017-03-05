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

    row_cnt = 1000
    data_n = 1000

    data_properties = dl.DataProperties(file_name, cols, class_col, row_cnt, data_n)

    return data_properties

def random_with_replacement(data, sample_n):
    sample = set()
    for i in range(sample_n):
        ind = int(random.random() * len(data))
        sample.add( data[ind] )

    return [list(s) for s in sample]

training_data = None
ranges = None

def get_classifier(training_data, ranges):
    logger.debug("Training classifier...")
    clf = ffc.FastFuzzyClassifier(random_with_replacement( training_data, int(0.6 * len(training_data)) ), ranges)
    clf.fit()

    return clf

def main():
    verification_data_perc = 0.1

    data_properties = poker_data_properties()
    data_loader_instance = dl.DataLoader(data_properties)
    data_loader_instance.set_logger(logger)
    data_loader_instance.load(shuffle=True)

    data = data_loader_instance.get_data()
    ranges = data_loader_instance.get_ranges()
    
    verification_data_n = int(verification_data_perc * len(data))
    training_data = data[:-verification_data_n]
    verification_data = data[-verification_data_n:]

    # best_features = range(10)

    # training_data = [  [ d[0][i] for i in best_features ], d[1] ] for d in training_data ]
    # verification_data = [ [ [ d[0][i] for i in best_features ], d[1] ] for d in verification_data ]
    # ranges = [ ranges[i] for i in best_features ]

    classifier_n = 8#int( len(data) ** 0.5 )
    classifiers = []
    with Pool(processes=4) as pool:
        classifiers = pool.starmap( get_classifier, [ (data, ranges) for i in range(classifier_n) ] )

    for t in training_data:
        logger.debug(str(t))
        avg = { "1": 0, "2" : 0 }
        for clf in classifiers:
            prediction = clf.predict(t[0])
            logger.debug("\t" + str(prediction))
            avg["1"] += prediction["1"]
            avg["2"] += prediction["2"]
        
        logger.debug("Average: " + str(avg))
        logger.debug("Winner: " + max(avg))
        input("Press enter to continue... ")


if __name__ == "__main__":
    set_logger()
    main()
