from treelib import Node, Tree
import numpy as np

import copy

import pyximport

pyximport.install()

from ..util import math_functions


class RandomFuzzyTree:
    NODE_NAME_LENGTH = 5
    NO_GOOD_CANDIDATES = {}
    MIN_GAIN_THRESHOLD = 0.000001

    def __init__(self,
                 feature_choice_n=5):
        self.is_fit = False
        self.feature_choice_n = feature_choice_n

    def fit(self, data, ranges, classes=(1, 2), a_cut=0.5):
        self.classes = classes
        self.a_cut = a_cut
        self.feature_n = data.shape[1] - 1

        self.tree = Tree()

        root_node = self.generate_root_node(data, ranges)
        self.tree.add_node(root_node)
        self.build_tree(root_node)

        self.tree.show()
        self.is_fit = True

    def generate_root_node(self, data, ranges):
        return Node("root",
                    data={"data": data,
                          "ranges": ranges,
                          "memberships": self.generate_root_memberships(data),
                          "f": lambda x: 1.0})

    def build_tree(self, node, lvl=0):
        if node.data["data"].shape[0] != 0:
            children = self.generate_best_children(node)

            if children != self.NO_GOOD_CANDIDATES:
                for c in children:
                    self.tree.add_node(c, node.identifier)

                for c in children:
                    self.build_tree(c, lvl + 1)

    def generate_best_children(self, node):
        features = self.generate_random_features()

        best_children_per_feature = {}
        for feature in features:
            print("\tFeature", feature)
            best_children = self.generate_best_feature_children(node, feature)
            if best_children != self.NO_GOOD_CANDIDATES:
                best_children_per_feature[feature] = best_children

        if best_children_per_feature == {}:
            best_feature_children = self.NO_GOOD_CANDIDATES
        else:
            best_feature = max(best_children_per_feature,
                               key=lambda x: self.gain(node, best_children_per_feature[x]))
            best_feature_children = best_children_per_feature[best_feature]

        return best_feature_children

    def generate_best_feature_children(self, node, feature):
        data = node.data["data"]
        sorted_inds = data[:, feature].argsort()
        data = data[sorted_inds]
        points = np.unique(data[1:-1, feature])

        lower, upper = node.data["ranges"][feature]

        last_point = data[0, feature]
        children_per_point = {}
        for p in points:
            diff = p - last_point
            if diff > (upper - lower) / 10 and diff > 10 and p - lower > 0.1 and upper - p > 0.1:
                children_at_point = self.generate_children_at_point(node, feature, p)
                if self.are_valid_children(children_at_point):
                    children_per_point[p] = \
                        children_at_point
                    last_point = p

        if children_per_point == {}:
            best_feature_children = self.NO_GOOD_CANDIDATES
        else:
            max_point = max(children_per_point,
                            key=lambda x: self.gain(node, children_per_point[x]))

            if self.gain(node, children_per_point[max_point]) < self.MIN_GAIN_THRESHOLD:
                best_feature_children = self.NO_GOOD_CANDIDATES
            else:
                best_feature_children = children_per_point[max_point]

        return best_feature_children

    def generate_random_features(self):
        features = np.random.choice(self.feature_n,
                                    self.feature_choice_n,
                                    replace=False)
        return features

    def is_root(self, memberships):
        return memberships is None

    def generate_root_memberships(self, data):
        item_n = data.shape[0]
        memberships = np.array([1 for _ in range(item_n)])

        return memberships

    def generate_children_at_point(self, node, feature, p):
        ranges = node.data["ranges"]
        lower, upper = ranges[feature]

        children = []

        child_ranges = copy.copy(ranges)
        # child_ranges[feature][1] = p
        left_triangular = math_functions.triangular(lower, p - lower)

        def child_f(item):
            return left_triangular(item[feature])

        children.append(self.generate_child_node(node,
                                                 child_ranges,
                                                 child_f))

        child_ranges = copy.copy(ranges)
        middle_triangular = math_functions.composite_triangular(p,
                                                                p - lower,
                                                                upper - p)

        def child_f(item):
            return middle_triangular(item[feature])

        children.append(self.generate_child_node(node,
                                                 child_ranges,
                                                 child_f))

        child_ranges = copy.copy(ranges)
        # child_ranges[feature][0] = p
        right_triangular = math_functions.triangular(upper, upper - p)

        def child_f(item):
            return right_triangular(item[feature])

        children.append(self.generate_child_node(node,
                                                 child_ranges,
                                                 child_f))

        return children

    def generate_child_node(self, node, child_ranges, child_f):
        data = node.data["data"]
        memberships = node.data["memberships"]

        child_f_values = np.apply_along_axis(child_f, 1, data)
        child_memberships = np.multiply(child_f_values, memberships)

        a_cut_passing_inds = (child_memberships >= self.a_cut).nonzero()
        child_memberships = child_memberships[a_cut_passing_inds]
        child_data = data[a_cut_passing_inds]

        return Node(data={"data": child_data,
                          "ranges": child_ranges,
                          "memberships": child_memberships,
                          "f": child_f})

    def gain(self, node, children):
        return 1

    def are_valid_children(self, children):
        non_zero_children = [c for c in children if c.data["data"].shape[0] != 0]
        non_zero_children_n = len(non_zero_children)

        return non_zero_children_n < 2
