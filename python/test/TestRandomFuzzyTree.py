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

        memberships = [1, 1, 1]

        feature = 2

        partition = FuzzyPartition()
        partition.f = triangular(2, 2)

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
            [[1, 2, 1, 1], 1],
            [[1, 2, 1.5, 2], 2],
            [[1, 2, 2, 3], 1],
            [[1, 2, 2, 3], 2],
            [[1, 2, 2.5, 3], 2],
            [[1, 2, 3, 3], 2],
        ]
        data = self.as_numpy(data)

        memberships = [1, 1, 1, 1, 1, 1]

        feature = 2

        p = 2
        L, U = 1, 3
        W_left = 2 * (p - L)
        W_middle = (U - L)
        W_right = 2 * (U - p)

        left_partition = FuzzyPartition()
        left_partition.f = math_functions.triangular(L, W_left)
        left_partition.properties = []

        middle_partition = FuzzyPartition()
        middle_partition.f = math_functions.triangular(p, W_middle)

        right_partition = FuzzyPartition()
        right_partition.f = math_functions.triangular(U, W_right)

        partitions = [left_partition,
                      middle_partition,
                      right_partition]

        self.tree.set_properties(partitions,
                                 data,
                                 feature,
                                 memberships)

        gain = self.tree.gain(partitions, memberships)
        self.assertAlmostEqual(gain, -0.69, 2)

    def testBestPartitioningTwoClusters(self):
        data = [
            [[1, 2, 1, 1], 1],
            [[1, 2, 1, 2], 2],
            [[1, 2, 1, 3], 1],
            [[1, 2, 3, 3], 2],
            [[1, 2, 3, 3], 2],
            [[1, 2, 3, 3], 2],
        ]
        data = self.as_numpy(data)

        memberships = [1, 1, 1, 1, 1, 1]

        feature = 2

        L = 1
        U = 3
        self.tree.ranges = [None, None, (L, U), None]

        best_partitioning = \
            self.tree.best_partitioning(feature, data, memberships)

        midpoint = L + (U - L) / 2

        left_partition = best_partitioning.partitions[0]
        self.assertAlmostEqual(left_partition.f(midpoint), 0, 2)

        middle_partition = best_partitioning.partitions[1]
        self.assertAlmostEqual(middle_partition.f(midpoint), 1, 2)

        right_partition = best_partitioning.partitions[2]
        self.assertAlmostEqual(right_partition.f(midpoint), 0, 2)

    def testBestPartitioningRegularData(self):
        data = [
            [[1, 2, 1, 1], 1],
            [[1, 2, 1.5, 2], 1],
            [[1, 2, 2.3, 3], 1],
            [[1, 2, 2.5, 3], 2],
            [[1, 2, 3, 3], 2],
        ]
        data = self.as_numpy(data)

        memberships = [1, 1, 1, 1, 1]

        feature = 2

        L = 1
        U = 3
        self.tree.ranges = [None, None, (L, U), None]

        best_partitioning = \
            self.tree.best_partitioning(feature, data, memberships)

        best_point = 1.5

        left_partition = best_partitioning.partitions[0]
        self.assertAlmostEqual(left_partition.f(best_point), 0, 2)

        middle_partition = best_partitioning.partitions[1]
        self.assertAlmostEqual(middle_partition.f(best_point), 1, 2)

        right_partition = best_partitioning.partitions[2]
        self.assertAlmostEqual(right_partition.f(best_point), 0, 2)

    @staticmethod
    def as_numpy(data):
        if len(data) == 0:
            return np.array([])
        x = np.array([d[0] for d in data])
        y = np.array([int(d[1]) for d in data])

        data = np.concatenate((x, np.array([y]).T), axis=1)

        return data
