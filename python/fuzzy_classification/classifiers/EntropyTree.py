from .Tree import Tree, Node, Split
import numpy as np
from math import log

class EntropyTree(Tree):
    
    def __init__(self,
                entropy_threshold=0.2,
                class_n=2):
        self.entropy_threshold = entropy_threshold
        self.class_n = class_n

    def _generate_proba_f(self, x, y):
        pass

    def _is_terminal_node(self, data):
        self._assert_1d_array(data)
        entropy = self._entropy(data)
        return entropy <= \
                self.entropy_threshold * self._max_entropy(data)

    def _generate_split(self, data, features):
        pass

    def _entropy(self, data):
        self._assert_1d_array(data)
        
        if self.class_n is None:
            raise ValueError(
                "You should supply the class_n parameter")

        histogram = np.histogram(data, bins=self.class_n)[0]
        p = histogram[ np.where(histogram != 0) ]
        p = 1 / data.shape[0] * p

        entropy = (-p*np.log2(p)).sum(axis=0)

        return entropy

    def _max_entropy(self, data):
        self._assert_1d_array(data)

        if self.class_n is None:
            raise ValueError(
                "You should supply the class_n parameter")
        else:
            max_entropy = log(self.class_n, 2)
            
            return max_entropy
