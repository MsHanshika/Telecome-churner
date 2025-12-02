import joblib
import numpy as np

class EnsembleModel:
    def __init__(self, models):
        self.models = models

    @classmethod
    def load(cls, path):
        models = joblib.load(path)
        return cls(models)

    def predict_proba(self, X):
        preds = np.column_stack([m.predict_proba(X)[:,1] for m in self.models])
        return preds.mean(axis=1)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
