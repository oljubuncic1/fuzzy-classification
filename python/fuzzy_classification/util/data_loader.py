import copy
from .dummy_logger import DummyLogger
import random
import logging



class DataProperties:
    def __init__(self, file_name, attribute_cols, class_col, row_cnt, data_n, filter_fun = None, transformation_fun = None, delimiter_char = ','):
        self.file_name = file_name
        self.attribute_cols = attribute_cols
        self.class_col = class_col
        self.row_cnt = row_cnt
        self.delimiter_char = delimiter_char
        self.filter_fun = filter_fun
        self.transformation_fun = transformation_fun
        self.data_n = data_n

class DataLoader:
    def __init__(self, data_properties):
        if not isinstance(data_properties, DataProperties):
            raise ValueError("Invalid file properties parameter type")

        self.data_properties = data_properties
        self.logger = DummyLogger()

    def set_logger(self, logger):
        logging.info("Setting logger...")
        self.logger = logger

    def example_from_line(self, line):
        parts = line.split(self.data_properties.delimiter_char)

        classification = parts[ self.data_properties.class_col ]
        parts = [ parts[i].strip() for i in self.data_properties.attribute_cols ]

        example = [parts, classification]

        return example

    def load_csv_data(self):
        self.logger.info("Loading data...")
        with open(self.data_properties.file_name) as f:
            content = head = [next(f) for x in range(self.data_properties.row_cnt)]
            lines = [x.strip() for x in content]
            data = [
                self.example_from_line(x)
                for x in lines
            ]

            self.data = data

            if self.data_properties.transformation_fun != None:
                f = self.data_properties.transformation_fun
                self.data = [ f(d) for d in self.data ]

            if self.data_properties.filter_fun != None:
                self.data = [ d for d in self.data if self.data_properties.filter_fun(d) ]
        
        self.data = [ ( tuple(d[0]), d[1] ) for d in self.data ]

        self.logger.info("Data loaded")

    def load_ranges(self):
        self.logger.info("Generating ranges...")        
        ranges = []
        for i in range( len(self.data[0][0]) ):
            ranges.append((
                min([float(e[0][i]) for e in self.data]),
                max([float(e[0][i]) for e in self.data])
            ))

        self.ranges = ranges

    def load(self, shuffle):
        if not shuffle:
            self.data_properties.row_cnt = self.data_properties.data_n
        self.load_csv_data()
        self.load_ranges()

        if shuffle:
            self.logger.info("Shuffling data...")
            random.shuffle(self.data)

        self.data = self.data[:self.data_properties.data_n]
    
    def get_data(self):
        return self.data
    
    def get_ranges(self):
        return self.ranges
