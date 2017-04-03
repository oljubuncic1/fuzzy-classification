import numpy as np
from math import log, sqrt

import pyximport

pyximport.install()
from ..util import math_functions

from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

from joblib import Parallel, delayed


class FuzzyNode:
    feature = None
    is_terminal=None
    # __slots__ = ['is_terminal',
    #              'classification',
    #              'feature'
    #              'partitioning']


class FuzzyPartitioning:
    __slots__ = ['partitions', 'gain']


class FuzzyPartition:
    __slots__ = ['f', 'node', 'properties']


class FuzzySetProperties:
    __slots__ = ['cardinality',
                 'entropy',
                 'data',
                 'memberships']


# noinspection PyAttributeOutsideInit,PyPropertyAccess,PyUnresolvedReferences
class RandomFuzzyTree:
    def __init__(self,
                 n_jobs=1,
                 p="sqrt",
                 terminal_n_threshold=10):

        self.n_jobs = n_jobs
        self.p = p
        self.is_fit = False
        self.terminal_n_threshold = terminal_n_threshold

    def fit(self, data, ranges, copy_data=False, classes=(1, 2)):
        self.classes = classes

        if copy_data:
            data = np.copy(data)

        self.ranges = ranges
        self.n_feature = self.count_features(data)

        if self.p == "sqrt":
            self.p = int(sqrt(self.n_feature))
        elif self.p == "log":
            self.p = int(log(self.n_feature, 2))
        elif self.p == "all":
            self.p = self.n_feature

        tree = self.build_tree(data, np.array([1.0 for d in data]))
        self.root = tree

        self.is_fit = True

    def predict(self, x):
        memberships = self.predict_memberships(x)
        return max(memberships)

    def predict_memberships(self, x):
        memberships = dict([(c, 0) for c in self.classes])
        self.forward_pass(memberships, x, self.root)

        return memberships

    def score(self, data):
        correct = 0
        for x in data:
            if self.predict(x[:-1]) == x[-1]:
                correct += 1

        return correct / data.shape[0]

    def build_tree(self, data, memberships, lvl=0):
        node = FuzzyNode()

        if self.is_terminal(data, memberships):
            node.is_terminal = True
            node.classification = self.classification(data, memberships)

            return node
        else:
            regular_features = []
            for i in range(len(self.ranges)):
                curr_range = self.ranges[i]
                if curr_range[0] != curr_range[1]:
                    regular_features.append(i)

            features = np.random.choice(regular_features,
                                        self.p,
                                        replace=False)

            feature_partitionings = {}
            for feature in features:
                feature_partitionings[feature] = \
                    self.best_partitioning(feature, data, memberships)

            node.feature = max(feature_partitionings,
                               key=lambda x: feature_partitionings[x].gain)
            node.partitioning = feature_partitionings[feature]

            for p in node.partitioning.partitions:
                p.node = self.build_tree(p.properties.data, p.properties.memberships, lvl + 1)

            return node

    def is_terminal(self, data, memberships):
        return np.sum(memberships) < 10

    def forward_pass(self,
                     result_memberships,
                     x,
                     node,
                     membership=1):

        if node.is_terminal:
            for c in self.classes:
                result_memberships[c] += node.classification[c]
        else:
            for partition in node.partitioning.partitions:
                next_membership = membership * partition.f(x[node.feature])
                next_node = partition.node

                self.forward_pass(result_memberships,
                             x,
                             next_node,
                             next_membership)

    @staticmethod
    def count_features(data):
        return data.shape[1] - 1

    def classification(self, data, memberships):
        classification_val = {}
        for c in self.classes:
            inds = (data[:, -1] == c).nonzero()[0]
            classification_val[c] = np.sum(memberships[inds])

        return classification_val

    def best_partitioning(self, feature, data, memberships):
        points = np.unique(data[:, feature])
        L, U = self.ranges[feature]

        point_partitionings = {}
        regular_point_occured = False
        for p in points:
            if p != L and p != U:
                regular_point_occured = True
                point_partitionings[p] = \
                    self.partitioning(data, feature, p, memberships)

        if not regular_point_occured:
            midpoint = L + (U - L) / 2
            max_partitioning = self.partitioning(data,
                                                 feature,
                                                 midpoint,
                                                 memberships)
        else:
            max_partitioning_key = max(point_partitionings,
                                       key=lambda x: point_partitionings[x].gain)

            max_partitioning = point_partitionings[max_partitioning_key]

        return max_partitioning

    def partitioning(self, data, feature, p, memberships):
        part = FuzzyPartitioning()

        L, U = self.ranges[feature]
        W_left = 2 * (p - L)
        W_middle_left = (p - L)
        W_middle_right = (U - p)
        W_right = 2 * (U - p)

        # TODO: generalize to more
        left_partition = FuzzyPartition()
        left_partition.f = math_functions.triangular(L,
                                                     W_left)
        left_partition.properties = []

        middle_partition = FuzzyPartition()
        middle_partition.f = \
            math_functions.composite_triangular(p,
                                                W_middle_left,
                                                W_middle_right)

        right_partition = FuzzyPartition()
        right_partition.f = math_functions.triangular(U,
                                                      W_right)

        part.partitions = [left_partition,
                           middle_partition,
                           right_partition]

        self.set_properties(part.partitions, data, feature, memberships)
        part.gain = self.gain(part.partitions, memberships)

        return part

    def set_properties(self, partitions, data, feature, memberships):
        for partition in partitions:
            prop = self._fuzzy_set_properties(data, feature, partition, memberships)
            partition.properties = prop

    def gain(self, partitions, memberships):
        data_cardinality = np.sum(memberships)
        if len(partitions) == 0:
            raise ValueError("Empty partitions")
        properties = [part.properties for part in partitions]
        gain_value = 0
        for prop in properties:
            gain_value -= prop.cardinality / data_cardinality * prop.entropy

        return gain_value

    def _fuzzy_set_properties(self, data, feature, partition, memberships):
        if data.shape.__contains__(0):
            raise ValueError("Empty array")

        membership_f = np.vectorize(partition.f)

        data_at_feature = np.copy(data[:, feature])
        set_memberships = membership_f(data_at_feature)

        set_memberships = np.multiply(memberships, set_memberships)
        cardinality = np.sum(set_memberships)

        entropy = self._fuzzy_entropy(data,
                                      set_memberships,
                                      cardinality)

        properties = FuzzySetProperties()
        properties.cardinality = cardinality
        properties.entropy = entropy

        non_zero_inds = np.nonzero(set_memberships)[0]
        set_data = data[non_zero_inds, :]
        set_memberships = set_memberships[non_zero_inds]

        properties.data = set_data
        properties.memberships = set_memberships

        return properties

    def _fuzzy_entropy(self, data, memberships, cardinality=None):
        if data.shape.__contains__(0):
            raise ValueError("Empty array")

        entropy = 0
        if cardinality is None:
            cardinality = np.sum(memberships)

        if cardinality != 0:
            for c in self.classes:
                inds = (data[:, -1] == c).nonzero()[0]
                memberships_at_inds = memberships[inds]
                proba = np.sum(memberships_at_inds) / cardinality
                if proba != 0:
                    entropy -= proba * log(proba, 2)

        return entropy

    def __str__(self):
        raise NotImplementedError()
