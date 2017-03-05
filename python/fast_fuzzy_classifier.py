from classifier import Classifier
from dummy_logger import DummyLogger
from operator import mul
from functools import reduce
import random
from multiprocessing import Pool
import math_functions as mf
import logging



class FastFuzzyClassifier(Classifier):
    def __init__(self, data, ranges, label_cnt=3, thread_n=4, positive_class='1', negative_class='2'):
        self.data = data
        self.ranges = ranges
        self.label_cnt = label_cnt
        self.logger = DummyLogger()
        self.thread_n = thread_n

        self.positive_class = positive_class
        self.negative_class = negative_class
    
    def __getstate__(self):
        """ This is called before pickling. """
        state = self.__dict__.copy()
        if 'logger' in state:
            del state['logger']
        return state

    def __setstate__(self, state):
        """ This is called while unpickling. """
        self.__dict__.update(state)


    def set_logger(self, logger):
        self.logger = logger

    def __labels(self, rng):
        L = rng[1] - rng[0]
        w = 2 * L / (self.label_cnt - 1)

        if w == 0:
            w = 1 # hacky, because division by zero

        return (
            [mf.triangular( rng[0] + i * w / 2.0, w, str(i) ) for i in range(self.label_cnt)]
        )

    def __generate_db(self):
        self.logger.info("Generating db...")
        self.db = [self.__labels(r) for r in self.ranges]

    def __rule(self, example):
        data = example[0]
        rw = 0.1
        rule = [[], example[1], rw]

        for i in range(len(self.db)):
            max_label = max(self.db[i], key=lambda x: x(data[i]))
            rule[0].append( max_label )
            for j in range(len(self.db[i])):
                f = self.db[i][j]

        return rule

    def __rule_str(self, r):
        r_str = ""
        for i in range(len(r[0])):
            r_str += r[0][i]('name')
        return r_str

    def __get_possible_labels(self, val, range):
        val = float(val)
        L = range[1] - range[0]
        if L == 0:
            L = 1
        half_cnt =  self.label_cnt - 1
        half_w = L / half_cnt

        ind = int((val - range[0]) / half_w)

        first = ind
        second = ind + 1

        if second >= self.label_cnt:
            first -= 1
            second -= 1

        return str(first), str(second)

    def get_rb(self):
        return self.rb

    def __generate_possible_rules(self, example, lvl = 0, curr = ""):
        # example is just list of atttributes here
        if lvl == len(self.ranges):
            return [curr]
        else:
            first, second = self.__get_possible_labels(example[lvl], self.ranges[lvl])

            return self.__generate_possible_rules(example, lvl + 1, curr + first) + \
                self.__generate_possible_rules(example, lvl + 1, curr + second)

    def __rule_from_rule_str(self, rule_str):
        r = [[], 0, 0]
        for i in range(len(rule_str)):
            curr_r = rule_str[i]
            r[0].append( self.db[i][int(curr_r)] )

        return r

    def __matching_degree(self, example, x):
        l = [  x[0][i](float(example[i])) for i in range(len(example)) ]
        
        dgr = reduce(mul, l, 1)
        return dgr

    def __my_classification(self, rule, examples):
        positive_sum = 0
        negative_sum = 0
        total_sum = 0
        for e in examples:
            curr_md = self.__matching_degree(e[0], rule)
            if e[1] == self.positive_class:
                positive_sum += curr_md
            elif e[1] == self.negative_class:
                negative_sum += curr_md
            total_sum += curr_md

        coefficient = abs(positive_sum - negative_sum) / total_sum

        if positive_sum > negative_sum:
            classification = self.positive_class
        else:
            classification = self.negative_class

        return [coefficient, classification]

    def __add_classifications(self):
        for r in self.rb:
            self.rb[r].extend(self.__my_classification(self.rb[r][0], self.rb[r][1]))

    def __generate_rb(self):
        self.logger.debug("Generating rb...")

        rb_map = {}
        for e in self.data:
            r = self.__rule(e)
            r_str = self.__rule_str(r)
            if r_str in rb_map:
                rb_map[r_str][0] = 1
                rb_map[r_str][2].append(e)
            else:
                rb_map[r_str] = [ 1, r, [e] ]

            possible_rules = self.__generate_possible_rules(e[0])
            for p in possible_rules:
                if p == r_str:
                    continue
                
                if p in rb_map:
                    rb_map[p][2].append(e)
                else:
                    rb_map[p] = [ 0, self.__rule_from_rule_str(p), [e] ]

        rb_map = dict( [ (r, [ rb_map[r][1], rb_map[r][2] ]) for r in rb_map if rb_map[r][0] != 0 ] )
        self.rb = rb_map

        self.__add_classifications()

    def fit(self):
        self.__generate_db()
        self.__generate_rb()

    def predict(self, example):
        # example is just list of atttributes here
        possible_rules = self.__generate_possible_rules(example)
        
        classification = { self.positive_class : 0.0, self.negative_class : 0.0 }
        for r in possible_rules:
            if r in self.rb:
                curr_md = self.__matching_degree(example, self.rb[r][0]) * self.rb[r][2]
                classification[ self.rb[r][3] ] += curr_md

        return classification

    def evaluate(self, verification_data):
        self.logger.info("Calculating accuracy...")

        self.logger.debug("Classifying...")
        with Pool(processes=self.thread_n) as pool:
            classifications = pool.map(self.predict, [ v[0] for v in verification_data])

        total = 0
        for i in range(len(verification_data)):
            verification = verification_data[i]
            classification = max(classifications[i])
            if classifications[i][classification] != 0.0 and classification == str(verification[1]):
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
