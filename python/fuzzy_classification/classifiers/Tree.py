from abc import ABC, ABCMeta, abstractmethod
from .Classifier import Classifier
import numpy as np
from math import sqrt


class Node:
    __slots__ = ['is_terminal',
                'proba_f',
                'split',
                'left',
                'right']

class Split:
    __slots__ = ['left_data',
                'right_data',
                'left_branch_criteria',
                'right_branch_criteria',
                'value',
                'feature']

    def free_data(self):
        left_data = None
        right_data = None

class Tree(Classifier, metaclass=ABCMeta):

    def __init__(self,
                n_jobs=1,
                is_random=True):
        super(Tree, self).__init__(n_jobs)
        self.is_random = is_random

    def fit(self, x, y=None):
        super(Tree, self).fit(x, y)
        features = list( range(self._feature_n(self.data)) )
        self.tree = self._generate_tree(self.data, features)

    def predict_proba(self, x):
        leaf_node = self._forward_pass(self.tree, x)

        return leaf_node.proba_f(x)

    @abstractmethod
    def _generate_proba_f(self, x, y):
        pass

    @abstractmethod
    def _is_terminal_node(self, dat, features):
        pass
    
    @abstractmethod
    def _generate_split(self, data, features):
        pass

    def _generate_tree(self, data, features, lvl=0):
        self._assert_numpy(data)

        node = Node()
        
        if not self._is_terminal_node(data[:, -1], features):
            if self.is_random:
                subset_n = int( sqrt(self.feature_n) )
                feature_subset = np.random.choice(features, 
                                                    subset_n)
                print("subset", feature_subset)
            else:
                feature_subset = features
            node.split = self._generate_split(data, feature_subset)
            features.remove(node.split.feature)
            node.is_terminal = False

            node.left = self._generate_tree(node.split.left_data, 
                                            features,
                                            lvl + 1)
            node.right = self._generate_tree(node.split.right_data, 
                                            features,
                                            lvl + 1)
            
            node.split.free_data()
        else:
            print("Terminal node", data[:,-1].astype(int))
            node.is_terminal = True
            node.proba_f = self._generate_proba_f(data[:,:-1], 
                                                data[:,-1])

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
