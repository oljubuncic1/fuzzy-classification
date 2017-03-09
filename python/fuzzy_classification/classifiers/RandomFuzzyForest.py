from ..util import math_functions as mf
from math import log
import random
import numpy as np



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

    def set_data(self, x, y):
        if y is None:
            self.y = np.array( [d[1] for d in x] )
            self.x = np.array( [d[0] for d in x] )
        else:
            self.x = np.array(x)
            self.y = np.array(y)

        self.y = self.y.astype(float)

    def predict(self, x):
        return

    def score(self, x, y):
        return

    def generate_tree(self, x, y, features_left=None, lvl = 0):
        if not self.is_terminal(y, features_left):
            if features_left is None:
                features_left = [i for i in range(len(x[0]))]

            node = Node()

            features = self.random_features(features_left)
            node.feature, node.cut = self.best_feature_and_cut(x, y, features)
            features.remove(node.feature)

            node.left_f = None
            node.right_f = None

            left_x, left_y, right_x, right_y = self.partitions(x, y, node)

            if len(left_y) == 0:
                node.classification = self.classification(right_y)
            elif len(right_y) == 0:
                node.classification = self.classification(left_y)
            else:
                node.left = self.generate_tree(left_x, left_y, features, lvl + 1)
                node.left = self.generate_tree(right_x, right_y, features, lvl + 1)

            return node
        else:
            node = Node()
            node.classification = self.classification(y)

            return node

    def is_terminal(self, y, features_left):
        entropy = mf.entropy(y)
        if entropy != 0:
            threshold = 0.8 * log(len(y), 2)
            is_terminal_entropy = entropy >= threshold
        else:
            is_terminal_entropy = True

        is_terminal_features = (features_left is not None and 
            len(features_left) == 0)

        return is_terminal_entropy or is_terminal_features 

    def classification(self, y):
        classes = {}

        for d in y:
            if d in classes:
                classes[d] += 1
            else:
                classes[d] = 0

        return max(classes)

    def partitions(self, x, y, node):
        print("Partitioning...")
        left_inds = [i for i in range(len(x)) if x[i][node.feature] < node.cut]
        right_inds = [i for i in range(len(x)) if x[i][node.feature] >= node.cut]

        left_x = [x[i] for i in left_inds]
        left_y = [y[i] for i in left_inds]

        right_x = [x[i] for i in right_inds]
        right_y = [y[i] for i in right_inds]

        return left_x, left_y, right_x, right_y

    def random_features(self, features_left):
        features = set()
        for i in range(self.feature_bag_n):
            rand_ind = int(random.random() * len(features_left))
            features.add(features_left[rand_ind])

        print(len(features))

        return list(features)

    def best_feature_and_cut(self, x, y, features):
        max_gain_and_cut = (0, None)
        max_feature = None
        for f in features:
            print("Feature ", f)
            curr_gain_and_cut = self.max_gain_and_cut(f, x, y)
            if curr_gain_and_cut[0] > max_gain_and_cut[0]:
                max_gain_and_cut = curr_gain_and_cut
                max_feature = f

        return max_feature, max_gain_and_cut[1]

    def max_gain_and_cut(self, f, x, y):
        x = x[:, f]

        arr = np.transpose( np.array([x, y]) )
        inds = arr[:, 0].argsort()
        arr = arr[inds]
        
        _max_gain_and_cut = (0, None)
        left_n, right_n = 0

        for i in range(len(arr)):

