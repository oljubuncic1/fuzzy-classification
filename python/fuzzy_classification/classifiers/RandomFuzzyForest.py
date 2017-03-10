from ..util import math_functions as mf
from math import log
import random
import numpy as np
import copy



class Node:
    left = None
    right = None

    left_f = None
    right_f = None

    feature = None
    cut = None

    classification = None


class RandomFuzzyForest:

    def __init__(self,
                n_cores=1,
                max_features=None,
                positive_class="1",
                negative_class="2"):
        if max_features is None:
            self.max_features = "sqrt"
        else:
            self.max_features = max_features

        self.n_cores = n_cores

    def fit(self, x, y):
        self.set_data(x, y)
        self.set_feature_bag_n(x)

        self.tree = self.generate_tree(self.x, self.y)

    def set_feature_bag_n(self, x):
        if self.max_features == "sqrt":
            self.feature_bag_n = int(len(x[0]) ** 0.5)
            # self.feature_bag_n = len(x[0])

    def set_data(self, x, y):
        if y is None:
            self.y = np.array( [d[1] for d in x] )
            self.x = np.array( [d[0] for d in x] )
        else:
            self.x = np.array(x)
            self.y = np.array(y)

        self.y = self.y.astype(float)

    def predict(self, x):
        return self.get_classification(x, self.tree)

    def score(self, x, y):
        correct_cnt = 0
        for i in range(len(x)):
            if self.predict(x[i]) == y[i]:
                correct_cnt += 1
        
        return correct_cnt / len(x)

    def get_classification(self, x, node):
        if node.classification is not None:
            return str(node.classification)
        else:
            val = x[node.feature]
            if val < node.cut:
                return self.get_classification(x, node.left)
            else:
                return self.get_classification(x, node.right)

    def generate_tree(self, x, y, features_left=None, lvl = 0):
        if not self.is_terminal(y, features_left):
            if features_left is None:
                features_left = [i for i in range(len(x[0]))]

            node = Node()

            features = self.random_features(features_left)
            result = self.best_feature_and_cut(x, y, features)
            node.feature = result[0]
            node.cut = result[1][1]
            left_x = result[1][2]
            left_y = result[1][3]
            right_x = result[1][4]
            right_y = result[1][5]
            features.remove(node.feature)

            node.left_f = None
            node.right_f = None

            if len(left_y) == 0:
                node.classification = self.classification(right_y)
            elif len(right_y) == 0:
                node.classification = self.classification(left_y)
            else:
                node.left = self.generate_tree(left_x, left_y, features, lvl + 1)
                node.right = self.generate_tree(right_x, right_y, features, lvl + 1)

            return node
        else:
            node = Node()
            node.classification = self.classification(y)

            return node

    def is_terminal(self, y, features_left):
        entropy = mf.entropy(y)
        threshold = log(y.shape[0], 2)
        if entropy != 0:
            is_terminal_entropy = (entropy >= threshold)
        else:
            is_terminal_entropy = True

        is_terminal_features = (features_left is not None and 
            len(features_left) == 0)

        return is_terminal_entropy or is_terminal_features 

    def classification(self, y):
        positive_cnt = len( np.where(y == 1) )
        if positive_cnt > len(y):
            return "1"
        else:
            return "2"
        
    def random_features(self, features_left):
        features = set()
        for i in range(self.feature_bag_n):
            rand_ind = int(random.random() * len(features_left))
            features.add(features_left[rand_ind])

        return list(features)

    def best_feature_and_cut(self, x, y, features):
        max_gain_and_cut = (0, None)
        max_feature = None
        for f in features:
            curr_gain_and_cut = self.max_gain_and_cut(f, x, y)
            if curr_gain_and_cut[0] > max_gain_and_cut[0]:
                max_gain_and_cut = curr_gain_and_cut
                max_feature = f

        return max_feature, max_gain_and_cut

    def max_gain_and_cut(self, f, x, y):
        my_x = np.array(x)
        my_x = my_x[:, f]

        arr = np.transpose( np.array([my_x, y]) )
        inds = arr[:, 0].argsort()
        arr = arr[inds]

        total_positive_n = np.where(arr[:, 1] == 1)[0].shape[0]
        total_negative_n = np.where(arr[:, 1] == 2)[0].shape[0]

        def entropy_gain(positive_n, negative_n):
            left_classes = [positive_n, negative_n]
            right_classes = [total_positive_n - positive_n, 
                total_negative_n - negative_n]
            return mf.entropy_by_nums(left_classes, arr.shape[0]) + \
                mf.entropy_by_nums(right_classes, arr.shape[0])
        
        _max_gain_and_cut = (0, None, -1)
        positive_n, negative_n = 0, 0

        for i in range(len(arr)):
            if arr[i][1] == 1:
                positive_n += 1
            else:
                negative_n += 1
            
            curr_gain = entropy_gain(positive_n, negative_n)
            if curr_gain > _max_gain_and_cut[0]:
                _max_gain_and_cut = curr_gain, arr[i][0], arr[i][0]


        return ( _max_gain_and_cut[0],
            _max_gain_and_cut[1], 
            x[ np.argwhere(arr[:, 0] <= _max_gain_and_cut[2]).flatten().tolist() ], 
            y[ np.argwhere(arr[:, 0] <= _max_gain_and_cut[2]).flatten().tolist() ],
            x[ np.argwhere(arr[:, 0] > _max_gain_and_cut[2]).flatten().tolist() ],
            y[ np.argwhere(arr[:, 0] > _max_gain_and_cut[2]).flatten().tolist() ] )
