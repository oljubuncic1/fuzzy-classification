from operator import mul
from functools import reduce
from collections import defaultdict
import math
import ann
from itertools import product
from multiprocessing import Pool
import math_functions as mf
from classifier import Classifier
from dummy_logger import DummyLogger
import logging



class ChiFuzzyClassifier(Classifier):
    def __init__(self, data, ranges, label_cnt = 3, thread_n = 1):
        self.data = data
        self.ranges = ranges
        self.label_cnt = label_cnt
        self.logger = DummyLogger()
        self.thread_n = thread_n

    def set_logger(logger):
        self.logger = logger

    def __labels(self, rng):
        L = rng[1] - rng[0]
        w = L / (self.label_cnt - 2)

        if w == 0:
            w = 1 # hacky, because division by zero

        return (
            [mf.triangular( rng[0] + i * w / 2.0, w, str(i) ) for i in range(self.label_cnt)]
        )

    def __generate_db(self):
        self.db = [self.__labels(r) for r in self.ranges]

    def __matching_degree(self, example, x):
        l = [  x[0][i](float(example[i])) for i in range(len(example)) ]
        return min(l)
        
        dgr = reduce(mul, l, 1)
        return dgr

    def rule(self, example):
        data = example[0]
        rw = 0.1
        rule = [[], example[1], rw]

        for i in range(len(self.db)):
            max_label = max(self.db[i], key=lambda x: x(data[i]))
            rule[0].append( max_label )

        rule[2] = self.__matching_degree(data, rule)
        
        return rule

    def __weight(self, rule):
        examples_in_class = [x for x in self.data if x[1] == rule[1]]
        examples_out_of_class = [x for x in self.data if x[1] != rule[1]]

        in_class_md = sum(self.__matching_degree(e[0], rule) for e in examples_in_class)
        out_of_class_md = sum(self.__matching_degree(e[0], rule) for e in examples_out_of_class)

        if in_class_md == 0 and out_of_class_md == 0:
            return 0

        weight = (in_class_md - out_of_class_md) / (in_class_md + out_of_class_md)

        return weight

    def __update_weights(self):
        for i in range( len(self.rb) ):
            self.rb[i][2] = self.__weight(self.rb[i])

    def update_weight(self, r):
        r[2] = self.__weight(r)
        return r

    def __rule_desc(self, rule):
        rule_desc_str = ""

        for f in rule[0]:
            rule_desc_str += f('name')
        rule_desc_str += rule[1]

        return rule_desc_str

    def __remove_duplicates(self):
        rule_map = defaultdict(list)
        for r in self.rb:
            rule_map[self.__rule_desc(r)].append(r)

        best_rules = []
        
        for k in rule_map:
            best_rules.append(max(rule_map[k], key=lambda x: x[2]))

        self.rb = best_rules

    def __generate_rb(self):
        self.rb = None
        with Pool(processes=self.thread_n) as pool:
            self.logger.debug("Creating rules...")
            self.rb = pool.map(self.rule, [ e for e in self.data ])
            # remove really same rules
            self.logger.debug("Updating weights...")
            self.rb = pool.map(self.update_weight, [ r for r in self.rb ] )

        self.logger.debug("Removing duplicates...")
        self.__remove_duplicates()

    def fit(self):
        self.__generate_db()
        self.__generate_rb()

    def predict(self, example):
        max_rule = max(self.rb, key=lambda x: self.__matching_degree(example, x) * x[2])
        
        return max_rule[1]

    def evaluate(self, verification_data):
        self.logger.info("Calculating accuracy...")

        self.logger.debug("Classifying...")
        with Pool(processes=self.thread_n) as pool:
            classifications = pool.map(self.predict, [ v[0] for v in verification_data])

        total = 0
        for i in range(len(verification_data)):
            verification = verification_data[i]
            if classifications[i] == str(verification[1]):
                total += 1

        accuracy = total / len(verification_data)
        self.logger.debug("Accuracy% " + str(accuracy))

        self.logger.debug("Class distribution")
        for c in set( [ x[1] for x in verification_data ] ):
            self.logger.debug(
                "\tData items with class " + c + " -> " + \
                    str( len( [x for x in verification_data if x[1] == c] ) )
            )

        return accuracy
