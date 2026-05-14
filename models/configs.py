"""
configs.py
==========
Defineix tots els models i el grid de feature engineering.
get_combos() retorna la llista plana de totes les combinacions a avaluar.
"""
from itertools import product
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor

try:
    from xgboost import XGBRegressor; HAS_XGB = True
except ImportError:
    HAS_XGB = False; print("[configs] xgboost no instal·lat — s'omet")

try:
    from lightgbm import LGBMRegressor; HAS_LGB = True
except ImportError:
    HAS_LGB = False; print("[configs] lightgbm no instal·lat — s'omet")


FE_GRID = {
    "poly":   [1, 2],
    "log1p":  [False, True],
    "rel_o3": [False, True],
    "scaler": [None, "standard"],
}

MODEL_DEFS = [
    (RandomForestRegressor,    "RF",    {"n_estimators":[100,300], "max_depth":[None,6], "random_state":[42]}),
    (GradientBoostingRegressor,"GBR",   {"n_estimators":[200], "max_depth":[3,5], "learning_rate":[0.05,0.1], "random_state":[42]}),
    (ExtraTreesRegressor,      "ET",    {"n_estimators":[200], "max_depth":[None,8], "random_state":[42]}),
    (Ridge,                    "Ridge", {"alpha":[0.1,1.0,10.0]}),
    (ElasticNet,               "EN",    {"alpha":[0.01,0.1,1.0], "l1_ratio":[0.3,0.7], "max_iter":[5000]}),
    (KNeighborsRegressor,      "KNN",   {"n_neighbors":[3,5,9], "weights":["uniform","distance"]}),
]
if HAS_XGB:
    MODEL_DEFS.append((XGBRegressor, "XGB",
        {"n_estimators":[200,400],"max_depth":[4,6],"learning_rate":[0.05,0.1],
         "subsample":[0.8],"random_state":[42],"verbosity":[0]}))
if HAS_LGB:
    MODEL_DEFS.append((LGBMRegressor, "LGB",
        {"n_estimators":[200,400],"num_leaves":[31,63],"learning_rate":[0.05,0.1],
         "random_state":[42],"verbose":[-1]}))

FE_KEYS = list(FE_GRID.keys())


def get_combos() -> list:
    """Retorna llista de dicts {class, base_name, params}."""
    out = []
    for fe_vals in product(*FE_GRID.values()):
        fe = dict(zip(FE_KEYS, fe_vals))
        for cls, name, hp_grid in MODEL_DEFS:
            for hp_vals in product(*hp_grid.values()):
                hp = dict(zip(hp_grid.keys(), hp_vals))
                out.append({"class": cls, "base_name": name, "params": {**fe, **hp}})
    return out


def model_hp(params: dict) -> dict:
    """Filtra les claus de FE i retorna només els hiperparàmetres del model."""
    return {k: v for k, v in params.items() if k not in FE_KEYS}


def build_id(base_name: str, params: dict) -> str:
    fe = (f"P{params.get('poly')}"
          f"_L{int(params.get('log1p',False))}"
          f"_R{int(params.get('rel_o3',False))}"
          f"_S{params.get('scaler') or 'none'}")
    hp = "_".join(f"{k}{v}" for k, v in params.items() if k not in FE_KEYS)
    return f"{base_name}_{fe}_{hp}".strip("_")
