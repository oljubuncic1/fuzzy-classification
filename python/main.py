import logging
import fuzzy_classification.util.data_loader as dl

from fuzzy_classification.classifiers.EntropyTree \
    import EntropyTree

import numpy as np

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
    file_name = "/home/faruk/workspace/thesis/data/covtype.data"
    cols = range(54)
    class_col = 54

    row_cnt =  int( 1* 10 ** 3 )
    data_n = int( 10 ** 4 )
    def filter_f(x):
        return x[1] == "1" or x[1] == "2"
    def trans_f(x):
        x[0] = [ int(d) for d in x[0] ]
        return x

    data_properties = dl.DataProperties(file_name, cols, class_col, row_cnt, data_n, filter_fun=filter_f, transformation_fun=trans_f)

    return data_properties

def as_numpy(data):
    x = np.array( [ d[0] for d in data ] )
    y = np.array( [ float(d[1]) for d in data ] )
    return x, y

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

    tree = EntropyTree(n_jobs=4, class_n=2)
    x, y = as_numpy(training_data)
    print("Fitting")
    tree.fit(x, y)
    x, y = as_numpy(verification_data)
    print( tree.score(x, y) )

if __name__ == "__main__":
    set_logger()
    main()
