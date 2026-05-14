"""
ensemble.py
===========
CrossValidationEnsemble: agrupa els models de cada fold CV
en un únic predictor (mitjana de prediccions).
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import transform_new


class CrossValidationEnsemble:
    def __init__(self, fold_models, params, fold_transformers):
        self.fold_models       = fold_models
        self.params            = params
        self.fold_transformers = fold_transformers

    def predict(self, X_raw) -> np.ndarray:
        preds = [m.predict(transform_new(X_raw, self.params, t))
                 for m, t in zip(self.fold_models, self.fold_transformers)]
        return np.mean(preds, axis=0)
