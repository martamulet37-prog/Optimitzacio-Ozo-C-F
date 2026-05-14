"""
features.py
===========
Ingenieria de features: poly, log1p, scaler, ràtios O3/coag.
Encapsula fit (sobre train) i transform (sobre qualsevol conjunt nou).
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures


def apply_fe(X_train: pd.DataFrame, X_val: pd.DataFrame, params: dict) -> tuple:
    """
    Fit sobre X_train, transforma X_train i X_val.
    Retorna (X_tr_arr, X_val_arr, transformers_dict).
    """
    t = {}
    Xtr, Xvl = X_train.copy(), X_val.copy()

    if params.get("rel_o3"):
        for df in [Xtr, Xvl]:
            df["O3_per_coag"]    = df["O3_dose"]   / (df["COAGULANT"] + 1e-6)
            df["coag_floc_ratio"] = df["COAGULANT"] / (df["FLOCULANT"] + 1e-6)

    if params.get("log1p"):
        cols = [c for c in Xtr.columns if Xtr[c].min() >= 0]
        Xtr[cols] = np.log1p(Xtr[cols])
        Xvl[cols] = np.log1p(Xvl[cols])
        t["log1p_cols"] = cols

    sc = {"standard": StandardScaler, "minmax": MinMaxScaler}.get(params.get("scaler"))
    if sc:
        scaler = sc()
        arr_tr = scaler.fit_transform(Xtr)
        arr_vl = scaler.transform(Xvl)
        t["scaler"] = scaler
    else:
        arr_tr, arr_vl = Xtr.values, Xvl.values

    if params.get("poly", 1) > 1:
        poly = PolynomialFeatures(degree=params["poly"], include_bias=False)
        arr_tr = poly.fit_transform(arr_tr)
        arr_vl = poly.transform(arr_vl)
        t["poly"] = poly

    return arr_tr, arr_vl, t


def transform_new(X: pd.DataFrame, params: dict, transformers: dict) -> np.ndarray:
    """Aplica transformers ja fitats a dades noves (planta real)."""
    X = X.copy()
    if params.get("rel_o3"):
        X["O3_per_coag"]    = X["O3_dose"]   / (X["COAGULANT"] + 1e-6)
        X["coag_floc_ratio"] = X["COAGULANT"] / (X["FLOCULANT"] + 1e-6)
    if params.get("log1p"):
        cols = [c for c in transformers.get("log1p_cols", []) if c in X.columns]
        X[cols] = np.log1p(X[cols].clip(lower=0))  # clip evita log1p(negatiu)
    arr = X.values
    if "scaler" in transformers:
        arr = transformers["scaler"].transform(arr)
    if "poly" in transformers:
        arr = transformers["poly"].transform(arr)
    return arr