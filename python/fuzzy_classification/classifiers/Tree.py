from abc import ABC, ABCMeta, abstractmethod
from .Classifier import Classifier
import numpy as np

class Node:
    __slots__ = ['is_terminal',
                'proba',
                'split',
                'left',
                'right']

class Split:
    __slots__ = ['left_data',
                'right_data',
                'left_branch_criteria',
                'right_branch_criteria',
                'value']

    def free_data(self):
        left_data = None
        right_data = None

class Tree(Classifier, metaclass=ABCMeta):

    def __init__(self,
                n_jobs=1):
        super(Classifier, self).__init__(n_jobs)

    def fit(self, x, y):
        super(Classifier, self).fit(x, y)
        data = self._combine(x, y)
        self.tree = self._generate_tree(data, 
                                        range(_feature_n(data)))

    def predict_proba(self, x):
        leaf_node = self._forward_pass(self.tree, x)

        return leaf_node.proba(x)

    @abstractmethod
    def _generate_proba_f(self, x, y):
        pass

    @abstractmethod
    def _is_terminal_node(self, data):
        pass
    
    @abstractmethod
    def _generate_split(self, data, features):
        pass

    def _combine(self, x, y):
        self._assert_numpy(x)
        self._assert_numpy(y)

    def _generate_tree(self, data, features):
        self._assert_numpy(data)

        node = Node()
        
        if not self.is_terminal_node(data):
            node.split = self._generate_split(data, features)
            node.is_terminal = False

            node.left = self._generate_tree(
                node.split.left_data)
            node.right = self._generate_tree(
                node.split.right_data)

            node.split.free_data()
        else:
            node.is_terminal = True
            node.proba = self._generate_proba(x, y)

        return node

    def _feature_n(self, data):
        self._assert_numpy(data)

        feature_n = data[0].shape[0] - 1
        return feature_n
    
    def _forward_pass(self, node, x):
        if node.is_terminal:
            return node
        else:
            if node.split.left_branch_criteria(x):
                return self._forward_pass(node.left, x)
            elif node.split.right_branch_criteria(x):
                return self._forward_pass(node.right, x)
