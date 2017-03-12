import unittest
from fuzzy_classification.classifiers.EntropyTree \
    import EntropyTree
import numpy as np


class testEntropyTree(unittest.TestCase):
    tree = None

    def setUp(self):
        self.tree = EntropyTree()

    def test_is_terminal_node_arguments(self):
        data = 1
        with self.assertRaises(ValueError):
            self.tree._is_terminal_node(data)

        data = [1]
        with self.assertRaises(ValueError):
            self.tree._is_terminal_node(data)

        data = [1, 2, 3]
        with self.assertRaises(ValueError):
            self.tree._is_terminal_node(data)

        with self.assertRaises(ValueError):
            data = np.array([])
            self.tree._is_terminal_node(data)

        with self.assertRaises(ValueError):
            data = np.array([[1, 2, 3], [4, 5, 6]])
            self.tree._is_terminal_node(data)

    def test_is_terminal_node(self):
        data = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 4])
        self.tree.class_n = 4
        self.assertFalse(self.tree._is_terminal_node(data))

        data = np.array([1, 1, 2, 2])
        self.tree.class_n = 2
        self.assertFalse(self.tree._is_terminal_node(data))

        data = np.array([1, 1, 1, 1])
        self.tree.class_n = 2
        self.assertTrue(self.tree._is_terminal_node(data))

    def test_entropy(self):
        data = np.array([1, 2])
        entropy = self.tree._entropy(data)
        self.assertAlmostEqual(entropy, 1)

        data = np.array([1, 1])
        entropy = self.tree._entropy(data)
        self.assertAlmostEqual(entropy, 0)

        data = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 4])
        self.tree.class_n = 4
        entropy = self.tree._entropy(data)
        self.assertAlmostEqual(entropy, 1.72, 2)

    def test_generate_proba_f(self):
        x = None
        y = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4])
        self.tree.class_n = 4
        proba_f = self.tree._generate_proba_f(x, y)

        proba = proba_f(1)
        should_proba =  {1: 3/10, 2: 3/10, 3: 3/10, 4: 1/10}

        for p in proba:
            if p in should_proba:
                self.assertAlmostEqual(proba[p], should_proba[p])
            else:
                self.assertTrue(False)
        
        self.assertEqual(proba_f(1), proba_f(2))

    def test_sorted_entropy_gain(self):
        self.tree.class_n = 4
        y = np.array([ 1, 1, 1, 2, 2, 2 ])

        self.assertAlmostEqual(self.tree._sorted_entropy_gain(y, 2), 0)

        self.tree.class_n = 2
        y = np.array([ 1, 2, 1, 1, 2, 2 ])

        self.assertAlmostEqual(self.tree._sorted_entropy_gain(y, 1), -2)

    def test_best_split(self):
        self.tree.class_n = 4
        data = np.array([ [1, 2, 3, 1],
                        [2, 3, 4, 1],
                        [3, 4, 5, 2],
                        [4, 5, 6, 2] ])

        best_split = self.tree._best_split(data, 0)
        self.assertEqual(best_split.left_data.shape[0], 2)
        self.assertEqual(best_split.value, 0)
        self.assertEqual(best_split.right_data.shape[0], 2)

        x = np.array([2, 3, 4])
        self.assertTrue(best_split.left_branch_criteria(x))

        x = np.array([2.01, 3, 4])
        self.assertFalse(best_split.left_branch_criteria(x))
        self.assertTrue(best_split.right_branch_criteria(x))

    def test_fit(self):
        self.class_n = 2
        data = np.array([ [1, 2, 3, 1],
                        [2, 3, 4, 1],
                        [3, 4, 5, 2],
                        [4, 5, 6, 2] ])
        self.tree.fit(data)
