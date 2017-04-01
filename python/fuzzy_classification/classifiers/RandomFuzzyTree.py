import numpy as np
from ..util import math_functions
from math import log


class FuzzyNode:
    __slots__ = ['is_terminal',
                 'classification',
                 'feature'
                 'partitioning']


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
                 p=5):

        self.n_jobs = n_jobs
        self.p = p
        self.is_fit = False

    def fit(self, data, ranges, copy_data=False, classes=(1, 2)):
        self.classes = classes

        if copy_data:
            data = np.copy(data)

        self.ranges = ranges
        self.n_feature = self.count_features(data)
        tree = self.build_tree(data, np.array([1.0 for d in data]))
        self.root = tree

        self.is_fit = True

    def build_tree(self, data, memberships):
        node = FuzzyNode()

        if self.is_terminal(data):
            node.is_terminal = True
            node.classification = self.classification(data, memberships)

            return node
        else:
            features = np.random.choice(range(self.n_feature),
                                        self.p,
                                        replace=False)

            feature_partitionings = {}
            for feature in features:
                feature_partitionings[feature] = \
                    self.best_partitioning(feature, data, memberships)

            node.feature = max(feature_partitionings,
                               key=lambda x: feature_partitionings[x].gain)
            node.partitioning = feature_partitionings[feature]

            for p in node.partitioning:
                p.node = self.build_tree(p.properties.data,
                                         p.properties.memberships)

            return node

    @staticmethod
    def is_terminal(data):
        return False

    @staticmethod
    def count_features(data):
        return data.shape[1] - 1

    def classification(self, data, memberships):
        pass

    def best_partitioning(self, feature, data, memberships):
        points = data[:, feature]

        point_partitionings = {}
        for p in points:
            point_partitionings[p] = \
                self.partitioning(data, feature, p, memberships)

        max_partitioning = max(point_partitionings,
                               key=lambda x: point_partitionings[x].gain)

        return point_partitionings[max_partitioning]

    def partitioning(self, data, feature, p, memberships):
        part = FuzzyPartitioning()

        L, U = self.ranges[feature]
        W = (p - L) / 2

        # TODO: generalize to more
        left_partition = FuzzyPartition()
        left_partition.f = math_functions.triangular(L, W)
        left_partition.properties = []

        middle_partition = FuzzyPartition()
        middle_partition.f = math_functions.triangular(p, W)

        right_partition = FuzzyPartition()
        right_partition.f = math_functions.triangular(U, W)

        part.partitions = [left_partition, middle_partition, right_partition]

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
        properties = [ part.properties for part in partitions ]
        gain_value = 0
        for prop in properties:
            gain_value -= prop.cardinality / data_cardinality * prop.entropy

        return gain_value

    def _fuzzy_set_properties(self, data, feature, partition, memberships):
        if data.shape.__contains__(0):
            raise ValueError("Empty array")
        membership_f = np.vectorize(partition.f)
        set_memberships = membership_f(data[:, feature])
        set_memberships = np.multiply(memberships, set_memberships)
        cardinality = np.sum(set_memberships)

        entropy = self._fuzzy_entropy(data, set_memberships)

        properties = FuzzySetProperties()
        properties.cardinality = cardinality
        properties.entropy = entropy

        non_zero_inds = np.nonzero(set_memberships)
        set_data = data[non_zero_inds, :]
        set_memberships = set_memberships[non_zero_inds]

        properties.data = set_data
        properties.memberships = set_memberships

        return properties

    def _fuzzy_entropy(self, data, memberships):
        if data.shape.__contains__(0):
            raise ValueError("Empty array")
        entropy = 0
        cardinality = np.sum(memberships)
        for c in self.classes:
            inds = (data[:, -1] == c).nonzero()[0]
            if not inds.shape.__contains__(0):
                proba = np.sum(memberships[inds]) / cardinality
                entropy -= proba * log(proba, 2)
        return entropy
