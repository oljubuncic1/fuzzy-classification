from .Tree import Tree, Node, Split
import numpy as np
from math import log
from multiprocessing import Pool


class EntropyTree(Tree):

    def __init__(self,
                n_jobs=1,
                class_n=2,
                entropy_threshold=0.2):
        self.entropy_threshold = entropy_threshold
        self.class_n = class_n
        super(EntropyTree, self).__init__(n_jobs)

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

    def _is_terminal_node(self, data, features):
        if len(features) == 0:
            return True
        
        self._assert_1d_array(data)
        if len(data) < 10:
            return True
        
        entropy = self._entropy(data)
        return entropy <= \
            self.entropy_threshold * self._max_entropy(data)

    def _generate_split(self, data, features):
        print(features)
        best_feature, best_gain = None, -float("inf")
        for f in features:
            best_split = self._best_split(data, f)
            gain = best_split.value
            if gain > best_gain:
                best_feature = f
                best_gain = gain

        best_split.feature = best_feature

        return best_split

    def _best_split(self, data, feature):
        self._assert_numpy(data)
        
        reduced_data = data[:, (feature, -1)]
        sorted_inds = reduced_data[:, 0].argsort()
        reduced_data = reduced_data[sorted_inds]
        cut_ind, max_entropy_gain = None, -float("inf")
        left_histogram = np.zeros(self.class_n)
        right_histogram = np.histogram(reduced_data[:, 1], 
                                        bins=self.class_n,
                                        range=(1, self.class_n))[0]

        for i in range(reduced_data.shape[0] - 1):
            if i > 1:
                histogram_ind = int(reduced_data[i][1]) - 1
                left_histogram[histogram_ind] += 1
                right_histogram[histogram_ind] -= 1
                entropy_gain = \
                    self._histogram_entropy_gain(left_histogram, 
                                                right_histogram)
                
                if entropy_gain > max_entropy_gain:
                    max_entropy_gain = entropy_gain
                    cut_ind = i

        best_split = Split()
        best_split.value = max_entropy_gain
        best_split.left_data = data[:cut_ind + 1,:]
        best_split.right_data = data[cut_ind + 1:,:]
        best_split.left_branch_criteria = \
            lambda x: x[feature] <= reduced_data[cut_ind, 0]
        best_split.right_branch_criteria = \
            lambda x: x[feature] > reduced_data[cut_ind, 0]

        return best_split
        
    def _histogram_entropy_gain(self, h1, h2):
        entropy1 = self._histogram_entropy(h1)
        entropy2 = self._histogram_entropy(h2)

        total_n = h1.shape[0] + h2.shape[0]

        return -(h1.shape[0] / total_n * entropy1 + \
                h2.shape[0] / total_n * entropy2)

    def _entropy(self, data):
        self._assert_1d_array(data)

        histogram = np.histogram(data, bins=self.class_n)[0]
        entropy = self._histogram_entropy(histogram)

        return entropy

    def _histogram_entropy(self, histogram):
        p = histogram[np.where(histogram != 0)]
        if p.shape[0] == 1:
            return -float("inf")
        p = 1 / histogram.sum() * p

        entropy = (-p * np.log2(p)).sum(axis=0)
        
        return entropy

    def _max_entropy(self, data):
        self._assert_1d_array(data)

        max_entropy = log(self.class_n, 2)
        return max_entropy
