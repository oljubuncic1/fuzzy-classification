from fuzzy_classification.classifiers.RandomFuzzyTree \
    import RandomFuzzyTree
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger("FuzzyEnsemble")


class FuzzyEnsemble:
    def __init__(self,
                 classifier_n=10,
                 classifier=RandomFuzzyTree):
        self.classifiers = []
        for i in range(classifier_n):
            self.classifiers.append(classifier())

    def fit(self, data, ranges, classes=(1, 2)):
        i = 1
        for c in self.classifiers:
            print("Fitting classifier %d" % i)
            i += 1

            inds = np.random.choice(range(data.shape[0]),
                                    size=int(0.8 * data.shape[0]),
                                    replace=True)
            classifier_data = data[inds,:]
            c.fit(classifier_data, ranges, classes=classes)

    def score(self, data):
        correct = 0
        for x in data:
            prediction = int(self.predict(x[:-1]))
            x_class = int(x[-1])

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
