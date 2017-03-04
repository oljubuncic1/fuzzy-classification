import logging
import data_loader as dl
import fast_fuzzy_classifier as ffc
import chi_fuzzy_classifier as cfc
import permutation_ga as pga



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

    row_cnt = 1000
    filter_fun = lambda x: x[1] in ['1', '2']

    data_properties = dl.DataProperties(file_name, cols, class_col, row_cnt, filter_fun)

    return data_properties

def kddcup_data_properties():
    file_name = "../data/kddcupfull.data"
    cols = [0, 4, 5, 9, 10, 12, 13, 14 , 15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 27, 28, 29,
        30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
    class_col = 41
    row_cnt = 1000

    data_properties = dl.DataProperties(file_name, cols, class_col, row_cnt)

    return data_properties

def covtype_data_properties():
    file_name = "../data/covtype.data"
    cols = range(54)
    class_col = 54

    row_cnt = 1000

    data_properties = dl.DataProperties(file_name, cols, class_col, row_cnt)

    return data_properties

def main():
    data_properties = poker_data_properties()
    data_loader_instance = dl.DataLoader(data_properties)
    data_loader_instance.load(shuffle=True)

    data = data_loader_instance.get_data()
    ranges = data_loader_instance.get_ranges()
    
    verification_data_perc = 0.1
    verification_data_n = int(verification_data_perc * len(data))
    training_data = data[:-verification_data_n]
    verification_data = data[-verification_data_n:]

    ffc_instance = ffc.FastFuzzyClassifier(training_data, ranges)
    ffc_instance.fit()
    ffc_acc = ffc_instance.evaluate(verification_data)

    cfc_instance = cfc.ChiFuzzyClassifier(data, ranges)
    cfc_instance.fit()
    cfc_acc = cfc_instance.evaluate(verification_data)

if __name__ == "__main__":
    set_logger()
    main()
