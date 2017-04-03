from fuzzy_classification.classifiers.RandomFuzzyTree \
    import RandomFuzzyTree
import numpy as np
from collections import defaultdict


class FuzzyEnsemble:
    def __init__(self,
                 classifier_n=10,
                 classifier=RandomFuzzyTree):
        self.classifiers = \
            [classifier() for i in range(classifier_n)]

    def fit(self, data, ranges):
        for c in self.classifiers:
            inds = np.random.choice(range(data.shape[0]), size=int(0.6 * data.shape[0]), replace=True)
            classifier_data = data[inds,:]
            c.fit(classifier_data, ranges)

    def score(self, data):
        correct = 0
        for x in data:
            prediction = self.predict(x[:-1])
            x_class = x[-1]

            if prediction == x_class:
                correct += 1

        return correct / data.shape[0]

    def predict(self, x):
        memberships = {}
        for c in self.classifiers:
            classifier_memberships = c.predict_memberships(x)
            for m in classifier_memberships:
                if m in memberships:
                    memberships[m] += classifier_memberships[m]
                else:
                    memberships[m] = classifier_memberships[m]

        return max(memberships, key=lambda k: memberships[k])