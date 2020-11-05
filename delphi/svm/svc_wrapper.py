from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC, SVC


class SVCWrapper(BaseEstimator):
    def __init__(self, probability: bool, kernel: str='linear', C: float=1.0, gamma: float=None):
        self.probability = probability
        self.kernel = kernel
        self.C = C
        self.gamma = gamma

        if kernel == 'linear':
            model = LinearSVC(random_state=42, class_weight='balanced', verbose=1, C=C)
            self.model = CalibratedClassifierCV(model) if self.probability else model
        else:
            self.model = SVC(random_state=42, class_weight='balanced', verbose=1, kernel=kernel, C=C, gamma=gamma)

    def fit(self, X, y, sample_weight=None):
        self.model.fit(X, y, sample_weight)

    def predict(self, X):
        return self.model.predict(X)

    def decision_function(self, X):
        return self.model.decision_function(X)
