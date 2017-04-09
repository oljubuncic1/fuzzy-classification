from treelib import Node, Tree
import numpy as np

from queue import Queue

import copy

import pyximport

pyximport.install()

from ..util import math_functions
from math import log


class RandomFuzzyTree:
    NO_GOOD_CANDIDATES = {}
    MIN_GAIN_THRESHOLD = 0.000001

    def __init__(self,
                 feature_choice_n=5,
                 max_depth=20):
        self.is_fit = False
        self.max_depth = max_depth
        self.feature_choice_n = feature_choice_n

    def fit(self, data, ranges, classes=(1, 2), a_cut=0.5):
        self.classes = classes
        self.a_cut = a_cut
        self.feature_n = data.shape[1] - 1

        self.tree = Tree()

        root_node = self.generate_root_node(data, ranges)
        self.tree.add_node(root_node)
        self.build_tree(root_node)

        # self.tree.show()
        self.is_fit = True

    def score(self, data):
        correct = 0
        for x in data:
            if self.predict(x[:-1]) == x[-1]:
                correct += 1

        return correct / data.shape[0]

    def predict(self, x):
        memberships = self.predict_memberships(x)
        print(memberships)

        return max(memberships, key=lambda x: memberships[x])

    def predict_memberships(self, x):
        memberships = {}
        for c in self.classes:
            memberships[c] = 0
        root = self.tree.nodes[self.tree.root]

        self.forward_pass(root, x, 1, memberships)

        return memberships

    def forward_pass(self, node, x, membership, memberships):
        membership *= node.data["f"](x)

        if node.is_leaf():
            node_classification = node.data["classification"]
            for c in self.classes:
                if c in node_classification:
                    memberships[c] += node_classification[c] * membership
        else:
            for child in node.fpointer:
                self.forward_pass(self.tree.nodes[child], x, copy.deepcopy(membership), memberships)

    def generate_root_node(self, data, ranges):
        return Node(tag="%s data_n: %d" % ("root", data.shape[0]),
                    data={"data": data,
                          "ranges": ranges,
                          "memberships": self.generate_root_memberships(data),
                          "f": lambda x: 1.0})

    def build_tree(self, node, lvl=0):
        # if lvl % 5 == 0:
        #     print("Building at level: ", lvl)
        # if lvl != 0:
        #     self.tree.show()

        frontier = Queue()
        frontier.put( (node, 0) )
        while frontier.qsize() != 0:
            node, lvl = frontier.get()
            if lvl < self.max_depth and node.data["data"].shape[0] != 0:
                children = self.generate_best_children(node)
                if children != self.NO_GOOD_CANDIDATES:
                    for c in children:
                        self.tree.add_node(c, node.identifier)

                    for c in children:
                        frontier.put( (c, lvl + 1) )

        return

        if lvl < self.max_depth and node.data["data"].shape[0] != 0:
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
        points = np.unique(data[:, feature])

        lower, upper = node.data["ranges"][feature]

        last_point = data[0, feature]
        children_per_point = {}
        for p in points:
            diff = p - last_point
            if diff > (upper - lower) / 50 and p - lower > 0.1 and upper - p > 0.1:
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

        self.append_left_child(children, feature, lower, node, p, ranges)
        self.append_mid_child(children, feature, lower, node, p, ranges, upper)
        self.append_right_child(children, feature, node, p, ranges, upper)

        return children

    def append_right_child(self, children, feature, node, p, ranges, upper):
        child_ranges = copy.deepcopy(ranges)
        diff = ranges[feature][1] - ranges[feature][0]
        child_ranges[feature][0] = p
        assert upper - p != 0
        right_triangular = math_functions.triangular(upper, 2.0 * (upper - p))
        child_f = lambda item: right_triangular(item[feature])
        child_name = "[feature: %d midpoint: %.2f range: (%.2f, %.2f)]" % (feature, upper, p, upper)
        child = self.generate_child_node(node, child_ranges, child_f, child_name)
        children.append(child)

    def append_mid_child(self, children, feature, lower, node, p, ranges, upper):
        child_ranges = copy.deepcopy(ranges)
        middle_triangular = math_functions.composite_triangular(p,
                                                                2 * (p - lower),
                                                                2 * (upper - p))
        child_f = lambda item: middle_triangular(item[feature])
        child_name = "[feature: %d midpoint: %.2f range: (%.2f, %.2f)]" % (feature, p, lower, upper)
        child = self.generate_child_node(node, child_ranges, child_f, child_name)
        children.append(child)

    def append_left_child(self, children, feature, lower, node, p, ranges):
        child_ranges = copy.deepcopy(ranges)
        diff = ranges[feature][1] - ranges[feature][0]
        child_ranges[feature][1] = p + diff
        assert p - lower != 0
        left_triangular = math_functions.triangular(lower, 2.0 * (p - lower))
        child_f = lambda item: left_triangular(item[feature])
        child_name = "[feature: %d midpoint: %.2f range: (%.2f, %.2f)]" % (feature, lower, lower, p)
        child = self.generate_child_node(node, child_ranges, child_f, child_name)
        children.append(child)

    def generate_child_node(self, node, child_ranges, child_f, name=None):
        data = node.data["data"]
        memberships = node.data["memberships"]

        child_f_values = np.apply_along_axis(child_f, 1, data)
        child_memberships = np.multiply(child_f_values, memberships)

        a_cut_passing_inds = (child_f_values > self.a_cut).nonzero()
        child_memberships = child_memberships[a_cut_passing_inds]
        child_data = copy.deepcopy(data[a_cut_passing_inds])

        child = Node(tag="%s data_n: %d" % (name, child_data.shape[0]),
                     data={"data": child_data,
                           "ranges": child_ranges,
                           "memberships": child_memberships,
                           "f": child_f})
        child.data["classification"] = \
            self.generate_node_classification(child, node)
        return child

    def gain(self, node, children):
        gain_value = self.fuzzy_entropy(node)

        for child in children:
            child_cardinality = self.fuzzy_cardinality(child)
            node_cardinality = self.fuzzy_cardinality(node)
            child_entropy = self.fuzzy_entropy(child)

            gain_value -= (child_cardinality / node_cardinality) * child_entropy

        return gain_value

    def are_valid_children(self, children):
        non_zero_children = [c for c in children if c.data["data"].shape[0] > 1]
        non_zero_children_n = len(non_zero_children)

        return non_zero_children_n >= 2

    def fuzzy_entropy(self, node):
        data = node.data["data"]
        if data.shape[0] == 0:
            entropy = 0
        else:
            entropy = 0
            cardinality = self.fuzzy_cardinality(node)
            memberships = node.data["memberships"]

            memberships_per_class = {}
            if cardinality != 0:
                for row in range(data.shape[0]):
                    cls = int(data[row,-1])
                    if cls in memberships_per_class:
                        memberships_per_class[cls] += memberships[row]
                    else:
                        memberships_per_class[cls] = memberships[row]

                for m in memberships_per_class:
                    ratio = memberships_per_class[m] / cardinality
                    entropy += ratio * log(ratio, 2)
                # for c in self.classes:
                #     class_indices = (data[:, -1] == c).nonzero()[0]
                #     memberships_at_inds = memberships[class_indices]
                #     proba = np.sum(memberships_at_inds) / cardinality
                #     if proba != 0:
                #         entropy -= proba * log(proba, 2)

        return entropy

    def fuzzy_cardinality(self, node):
        memberships = node.data["memberships"]

        return np.sum(memberships)

    def generate_node_classification(self, node, parent=None):
        data = node.data["data"]
        memberships = node.data["memberships"]

        # if data.shape[0] == 0:
        #     assert parent is not None
        # return self.generate_node_classification(parent)
        # else:
        classification_val = {}

        for c in self.classes:
            inds = (data[:, -1] == c).nonzero()[0]
            classification_val[c] = np.sum(memberships[inds])

        return classification_val
