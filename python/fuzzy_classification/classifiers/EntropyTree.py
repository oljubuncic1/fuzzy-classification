from .Tree import Tree, Node, Split
import numpy as np
from math import log


class EntropyTree(Tree):

    def __init__(self,
                class_n=2,
                entropy_threshold=0.2):
        self.entropy_threshold = entropy_threshold
        self.class_n = class_n

    def _generate_proba_f(self, x, y):
        self._assert_1d_array(y)

        histogram = np.histogram(y, 
                                bins=self.class_n, 
                                range=(1, self.class_n))[0]
        probabilities = histogram
        probabilities = 1 / y.shape[0] * probabilities

        probability_map = [(i + 1, probabilities[i])
                           for i in range(len(probabilities))]
        
        def proba(x):
            return dict(probability_map)

        return proba

    def _is_terminal_node(self, data):
        self._assert_1d_array(data)
        entropy = self._entropy(data)
        return entropy <= \
            self.entropy_threshold * self._max_entropy(data)

    def _generate_split(self, data, features):
        best_feature, best_gain = None, 0
        for f in features:
            best_split = self._best_split(data, f)
            if best_split.value > best_gain:
                best_feature = f
                best_gain = gain

        features.remove(best_feature)

        return best_split

    def _best_split(self, data, feature):
        raise NotImplementedError()

    def _entropy(self, data):
        self._assert_1d_array(data)

        histogram = np.histogram(data, bins=self.class_n)[0]
        p = histogram[np.where(histogram != 0)]
        p = 1 / data.shape[0] * p

        entropy = (-p * np.log2(p)).sum(axis=0)

        return entropy

    def _max_entropy(self, data):
        self._assert_1d_array(data)

        max_entropy = log(self.class_n, 2)
        return max_entropy
