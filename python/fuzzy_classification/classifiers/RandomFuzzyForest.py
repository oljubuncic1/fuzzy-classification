from ..util import math_functions as mf
from math import log
import random



class Node:
    left = None
    right = None

    left_f = None
    right_f = None

    feature = None
    cut = None

    classification = None


class RandomFuzzyForest:

    def __init__(
        self,
        n_cores=1,
        max_features=None,
        positive_class="1",
        negative_class="2"
    ):
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
            self.y = [d[1] for d in x]
            self.x = [d[0] for d in x]
        else:
            self.x = x
            self.y = y

    def predict(self, x):
        return

    def score(self, x, y):
        return

    def generate_tree(self, x, y, features_left=None):
        if not self.is_terminal(y):
            if features_left is None:
                features_left = [i for i in range(len(x[0]))]

            node = Node()

            features = self.random_features(features_left)
            node.feature, node.cut = self.best_feature_and_cut(x, y, features)
            features.remove(node.feature)

            node.left_f = None
            node.right_f = None

            left_x, left_y, right_x, right_y = self.partitions(x, y, node)

            node.left = self.generate_tree(left_x, left_y, features)
            node.left = self.generate_tree(right_x, right_y, features)

            return node
        else:
            node = Node()
            node.classificatin = classification(y)

            return node

    def is_terminal(self, y):
        entropy = mf.entropy(y)
        threshold = 0.8 * log(len(y), 2)

        return entropy >= threshold

    def classification(self, y):
        classes = {}

        for d in y:
            if d in classes:
                classes[d] += 1
            else:
                classes[d] = 0

        return max(classes)

    def partitions(self, x, y, node):
        left_inds = [i for i in range(len(x)) if d[i][node.feature] < node.cut]
        right_inds = [i for i in range(len(x)) if d[i][node.feature] >= node.cut]

        left_x = [x[i] for i in left_inds]
        left_y = [y[i] for i in left_inds]

        right_x = [x[i] for i in left_inds]
        right_y = [y[i] for i in left_inds]

        return left_x, left_y, right_x, right_y

    def random_features(self, features_left):
        features = set()
        for i in range(self.feature_bag_n):
            rand_ind = int(random.random() * len(features_left))
            features.add(features_left[rand_ind])

        return list(features)

    def best_feature_and_cut(self, x, y, features):
        feature_cuts = {}
        for f in features:
            feature_cuts[f] = self.max_gain_and_cut(f, x, y)

        max_feature = max(feature_cuts, key=lambda x: feature_cuts[x][0])
        max_cut = feature_cuts[max_feature][1]

        return max_feature, max_cut

    def max_gain_and_cut(self, f, x, y):
        def entropy_gain(f, x, y, cut):
            ind = 0
            while ind < len(x) and cut != x[ind][f]:
                ind = ind + 1

            return mf.entropy([y[i] for i in range(len(y)) if i <= ind]) + \
                mf.entropy([y[i] for i in range(len(y)) if i > ind])

        max_cut = max(x, key=lambda d: entropy_gain(f, x, y, d[f]))

        return entropy_gain(f, x, y, max_cut), max_cut
