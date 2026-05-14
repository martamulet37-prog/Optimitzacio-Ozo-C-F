"""
CrossValidationEnsemble
-----------------------
Bundles all fold models from a CV run into a single predictor.
Prediction = mean across all fold models (ensemble averaging).
"""
import sys
import os
import numpy as np

# Resolve src.features regardless of where the script is called from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import transform_new


class CrossValidationEnsemble:
    """
    Parameters
    ----------
    fold_models       : list of fitted MultiOutputRegressor (one per CV fold)
    params            : FE + model hyperparameter dict used during training
    fold_transformers : list of transformer dicts (one per fold)
    """

    def __init__(self, fold_models: list, params: dict, fold_transformers: list):
        self.fold_models       = fold_models
        self.params            = params
        self.fold_transformers = fold_transformers

    def predict(self, X_raw) -> np.ndarray:
        """Average predictions of all fold models on raw (untransformed) X."""
        preds = []
        for model, t_set in zip(self.fold_models, self.fold_transformers):
            X_t = transform_new(X_raw, self.params, t_set)
            preds.append(model.predict(X_t))
        return np.mean(preds, axis=0)   # shape (n_samples, n_targets)
