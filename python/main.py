import logging
import fuzzy_classification.util.data_loader as dl
from fuzzy_classification.classifiers.FuzzyEnsemble import FuzzyEnsemble
from sklearn.ensemble import RandomForestClassifier
from fuzzy_classification.classifiers.RandomFuzzyTree \
    import RandomFuzzyTree

import numpy as np
import random

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

    row_n = int(10 ** 3)
    data_n = int(10 ** 3)

    def trans_f(x):
        if x[1] != '1':
            x[1] = '2'
        return x

    data_properties = dl.DataProperties(file_name, cols, class_col, row_n, data_n, transformation_fun=trans_f)

    return data_properties


def kddcup_data_properties():
    file_name = "../data/kddcupfull.data"
    cols = [0, 4, 5, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 27, 28, 29,
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

    row_cnt = int(15 * 10 ** 3)
    data_n = int(15 * 10 ** 3)

    def filter_f(x):
        return x[1] == "1" or x[1] == "2"

    def trans_f(x):
        x[0] = [int(d) for d in x[0]]
        return x

    data_properties = dl.DataProperties(file_name, cols, class_col, row_cnt, data_n, filter_fun=filter_f,
                                        transformation_fun=trans_f)

    return data_properties


def haberman_data_properties():
    file_name = \
        "/home/faruk/workspace/thesis/data/haberman.dat"
    cols = range(3)
    class_col = 3

    row_cnt = int(306)
    data_n = int(306)

    def trans_f(x):
        x[0] = [int(d) for d in x[0]]
        return x

    data_properties = dl.DataProperties(file_name,
                                        cols,
                                        class_col,
                                        row_cnt,
                                        data_n,
                                        transformation_fun=trans_f)

    return data_properties


def mamographic_data_properties():
    file_name = \
        "/home/faruk/workspace/thesis/data/mamographic.dat"
    cols = range(5)
    class_col = 5

    row_cnt = int(961)
    data_n = int(961)

    def filter_f(x):
        return not "?" in x[0] and not x[1] == "?"

    def trans_f(x):
        if filter_f(x):
            x[0] = [float(d) for d in x[0]]
            x[1] = str(int(x[1]) + 1)
        return x

    data_properties = dl.DataProperties(file_name,
                                        cols,
                                        class_col,
                                        row_cnt,
                                        data_n,
                                        transformation_fun=trans_f,
                                        filter_fun=filter_f)

    return data_properties


def iris_data_properties():
    file_name = \
        "/home/faruk/workspace/thesis/data/iris.dat"
    cols = range(4)
    class_col = 4

    row_cnt = int(150)
    data_n = int(150)

    def trans_f(x):
        x[0] = [float(d) for d in x[0]]
        if x[1] == "Iris-setosa":
            x[1] = "1"
        elif x[1] == "Iris-versicolor":
            x[1] = "2"
        else:
            x[1] = "3"
        return x

    data_properties = dl.DataProperties(file_name,
                                        cols,
                                        class_col,
                                        row_cnt,
                                        data_n,
                                        transformation_fun=trans_f)

    return data_properties


def contraceptive_data_properties():
    file_name = \
        "/home/faruk/workspace/thesis/data/contraceptive.dat"
    cols = range(9)
    class_col = 9

    row_cnt = int(1473)
    data_n = int(1473)

    def trans_f(x):
        x[0] = [float(d) for d in x[0]]
        return x

    data_properties = dl.DataProperties(file_name,
                                        cols,
                                        class_col,
                                        row_cnt,
                                        data_n,
                                        transformation_fun=trans_f)

    return data_properties


def segmentation_data_properties():
    file_name = \
        "/home/faruk/workspace/thesis/data/segmentation.dat"
    cols = list(range(1, 20))
    cols.remove(3)
    class_col = 0

    row_cnt = int(2100)
    data_n = int(1000)

    def trans_f(x):
        x[0] = [float(d) for d in x[0]]
        if x[1] == "BRICKFACE":
            x[1] = "1"
        elif x[1] == "SKY":
            x[1] = "2"
        elif x[1] == "FOLIAGE":
            x[1] = "3"
        elif x[1] == "CEMENT":
            x[1] = "4"
        elif x[1] == "WINDOW":
            x[1] = "5"
        elif x[1] == "PATH":
            x[1] = "6"
        elif x[1] == "GRASS":
            x[1] = "7"

        return x

    data_properties = dl.DataProperties(file_name,
                                        cols,
                                        class_col,
                                        row_cnt,
                                        data_n,
                                        transformation_fun=trans_f)

    return data_properties


def as_numpy(data):
    x = np.array([d[0] for d in data])
    y = np.array([int(d[1]) for d in data])

    data = np.concatenate((x, np.array([y]).T),
                          axis=1)

    return data


def main():
    verification_data_perc = 0.1

    data_properties = segmentation_data_properties()
    data_loader_instance = dl.DataLoader(data_properties)
    data_loader_instance.set_logger(logger)
    data_loader_instance.load(shuffle=True)

    data = data_loader_instance.get_data()
    print("Data size:", len(data))
    ranges = data_loader_instance.get_ranges()

    verification_data_n = int(verification_data_perc * len(data))
    training_data = data[:-verification_data_n]
    verification_data = data[-verification_data_n:]

    np_training_data = as_numpy(training_data)
    np_verification_data = as_numpy(verification_data)

    rf = RandomForestClassifier(n_jobs=4, max_features="log2", criterion="entropy", n_estimators=10, bootstrap=True)
    rf.fit([t[0] for t in training_data], [t[1] for t in training_data])
    score = rf.score([t[0] for t in verification_data], [t[1] for t in verification_data])
    print("Crisp Forest score", score)

    ff = FuzzyEnsemble(classifier_n=8)
    ranges = [list(r) for r in ranges]
    # for i in range(len(ranges)):
        # diff = ranges[i][1] - ranges[i][0]
        # ranges[i][0] = min(0, ranges[i][0])
        # ranges[i][1] += diff / 10
    ff.fit(np_training_data, ranges, classes=[1, 2, 3, 4, 5, 6, 7])
    print("Score: ", ff.score(np_verification_data))

    verification_data_perc = {}
    for v in verification_data:
        if v[1] in verification_data_perc:
            verification_data_perc[v[1]] += 1
        else:
            verification_data_perc[v[1]] = 0

    print(verification_data_perc)

if __name__ == "__main__":
    set_logger()
    main()
