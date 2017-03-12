from abc import ABC, ABCMeta, abstractmethod
import numpy as np



class Classifier(metaclass=ABCMeta):
    
    def __init__(self,
                n_jobs=1):
        self.n_jobs = n_jobs
    
    @abstractmethod
    def fit(self, x, y):
        self._try_parse_input(x, y)
        self.is_fit = True
        self.feature_n = self.data.shape[1] - 1

    @abstractmethod
    def predict_proba(self, x):
        self._assert_numpy(x)

    def predict(self, x):
        if not self.is_fit:
            raise AssertionError("Classifier not fit to any data") 

        proba = self.predict_proba(x)
        return max(proba)

    def score(self, x, y):
        self._try_parse_input(x, y)

        x = self.data[:,:-1]
        y = self.data[:, -1].astype(int)

        predictions = np.apply_along_axis(self.predict, 1, x)

        total_elements = y.shape[0]
        total_correct = \
            np.count_nonzero(np.equal(predictions, np.transpose(y)))
        return total_correct / x.shape[0]

    def _try_parse_input(self, x, y):
        if y is None:
            self._assert_numpy(x)
            self.data = x
        else:
            self._assert_numpy(x)
            self._assert_numpy(y)
            
            self.data = self._combine(x, y)

    def _combine(self, x, y):
        self._assert_numpy(x)
        self._assert_numpy(y)

        return np.concatenate( (x, np.array([y]).T), 
                                axis=1 )

    def _assert_numpy(self, data):
        if not isinstance(data, np.ndarray):
            raise ValueError("You should supply numpy array")

    def _assert_1d_array(self, data):
        self._assert_numpy(data)

        if data.ndim > 1:
            raise ValueError(
                "You should supply 1D array of classes")
        elif data.shape[0] == 0:
            raise ValueError(
                "You should supply non-empty array")

    is_fit = False
