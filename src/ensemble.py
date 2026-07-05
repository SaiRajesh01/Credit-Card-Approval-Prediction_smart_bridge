import numpy as np

class EnsembleClassifier:
    """Ensemble model that averages prediction probabilities of multiple classifiers."""
    
    def __init__(self, estimators):
        self.estimators = estimators

    def fit(self, X, y):
        # Already fitted individually
        pass

    def predict_proba(self, X):
        probs = [model.predict_proba(X) for model in self.estimators]
        return np.mean(probs, axis=0)

    def predict(self, X):
        # Standard binary threshold prediction
        prob = self.predict_proba(X)[:, 1]
        return (prob >= 0.5).astype(int)
