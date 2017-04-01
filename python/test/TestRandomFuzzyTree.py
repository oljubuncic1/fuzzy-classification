import unittest
import numpy as np

from math import log

from fuzzy_classification.util import math_functions
from fuzzy_classification.util.math_functions import triangular
from fuzzy_classification.classifiers.RandomFuzzyTree \
    import RandomFuzzyTree, FuzzyPartition


# noinspection PyPropertyAccess
class TestRandomFuzzyTree(unittest.TestCase):
    tree = None

    def setUp(self):
        self.tree = RandomFuzzyTree()
        self.tree.classes = (1, 2)

    def testCreation(self):
        pass

    def testFuzzyEntropyEmptyArray(self):
        data = []
        data = self.as_numpy(data)
        memberships = []

        with self.assertRaises(ValueError):
            self.tree._fuzzy_entropy(data, memberships)

    def testFuzzyEntropyRegularArray(self):
        data = [
            [[1, 2, 3, 1], 1],
            [[1, 2, 3, 2], 2],
            [[1, 2, 3, 3], 2],
        ]
        data = self.as_numpy(data)
        memberships = np.array([0.1, 0.2, 0.2])
        self.tree.classes = (1, 2)

        fuzzy_entropy = \
            self.tree._fuzzy_entropy(data, memberships)

        self.assertAlmostEqual(fuzzy_entropy, 0.722, 3)

    def testFuzzyEntropyUniformArray(self):
        data = [
            [[1, 2, 3, 1], 1],
            [[1, 2, 3, 2], 1],
            [[1, 2, 3, 3], 1],
        ]
        data = self.as_numpy(data)
        memberships = np.array([0.1, 0.2, 0.2])

        fuzzy_entropy = \
            self.tree._fuzzy_entropy(data, memberships)

        self.assertAlmostEqual(fuzzy_entropy, 0, 2)

    def testFuzzySetPropertiesEmptyArray(self):
        data = [

        ]
        data = self.as_numpy(data)
        partition = FuzzyPartition()
        feature = 0
        memberships = []

        with self.assertRaises(ValueError):
            self.tree._fuzzy_set_properties(data,
                                            feature,
                                            partition,
                                            memberships)

    def testFuzzySetPropertiesUniformArray(self):
        data = [
            [[1, 2, 3, 1], 1],
            [[1, 2, 3, 2], 1],
            [[1, 2, 3, 3], 1],
        ]
        data = self.as_numpy(data)

        feature = 2

        partition = FuzzyPartition()
        partition.f = triangular(3, 1)

        memberships = [1, 1, 1]

        fuzzy_set_properties = \
            self.tree._fuzzy_set_properties(data,
                                            feature,
                                            partition,
                                            memberships)

        cardinality = fuzzy_set_properties.cardinality
        self.assertAlmostEqual(cardinality, 3, 2)

        entropy = fuzzy_set_properties.entropy
        self.assertAlmostEqual(entropy, 0, 2)



    def testFuzzySetPropertiesRegularArray(self):
        data = [
            [[1, 2, 2.5, 1], 1],
            [[1, 2, 1.5, 2], 1],
            [[1, 2, 2, 3], 2],
        ]
        data = self.as_numpy(data)

        feature = 2

        partition = FuzzyPartition()
        partition.f = triangular(2, 2)

        memberships = [1, 1, 1]

        fuzzy_set_properties = \
            self.tree._fuzzy_set_properties(data,
                                            feature,
                                            partition,
                                            memberships)

        cardinality = fuzzy_set_properties.cardinality
        self.assertAlmostEqual(cardinality, 2, 2)

        entropy = fuzzy_set_properties.entropy
        self.assertAlmostEqual(entropy, -log(0.5, 2), 2)

    def testGainEmptyPartitions(self):
        partitions = []
        memberships = []

        with self.assertRaises(ValueError):
            self.tree.gain(partitions, memberships)

    def testGainRegularPartitions(self):
        data = [
            [[1, 2, 2.5, 1], 1],
            [[1, 2, 1.5, 2], 1],
            [[1, 2, 2, 3], 2],
        ]
        data = self.as_numpy(data)

        feature = 2

        p = 1.5
        L, U = 1, 2
        W = (p - L) / 2

        left_partition = FuzzyPartition()
        left_partition.f = math_functions.triangular(L, W)
        left_partition.properties = []

        middle_partition = FuzzyPartition()
        middle_partition.f = math_functions.triangular(p, W)

        right_partition = FuzzyPartition()
        right_partition.f = math_functions.triangular(U, W)

        partitions = [left_partition,
                      middle_partition,
                      right_partition]
        memberships = []

        self.tree.set_properties(partitions,
                                 data,
                                 feature,
                                 memberships)

        self.tree.gain(partitions, memberships)

    @staticmethod
    def as_numpy(data):
        if len(data) == 0:
            return np.array([])
        x = np.array([d[0] for d in data])
        y = np.array([int(d[1]) for d in data])

        data = np.concatenate((x, np.array([y]).T),
                              axis=1)

        return data
