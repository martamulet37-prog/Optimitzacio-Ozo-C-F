"""
All model classes and hyperparameter grids used in the benchmark.
get_all_combinations() returns a flat list of all model x FE configs to evaluate.
"""
from itertools import product
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[base_configs] xgboost not installed — skipping XGB models")

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[base_configs] lightgbm not installed — skipping LGB models")


# ── Feature-engineering grid ──────────────────────────────────────────────────
FE_GRID = {
    "poly":   [1, 2],
    "log1p":  [False, True],
    "rel_o3": [False, True],
    "scaler": [None, "standard"],
}

# ── Model definitions: (class, short_name, hyperparam_grid) ──────────────────
MODEL_DEFS = [
    (
        RandomForestRegressor, "RF",
        {"n_estimators": [100, 300], "max_depth": [None, 6], "random_state": [42]},
    ),
    (
        GradientBoostingRegressor, "GBR",
        {"n_estimators": [200], "max_depth": [3, 5],
         "learning_rate": [0.05, 0.1], "random_state": [42]},
    ),
    (
        ExtraTreesRegressor, "ET",
        {"n_estimators": [200], "max_depth": [None, 8], "random_state": [42]},
    ),
    (
        Ridge, "Ridge",
        {"alpha": [0.1, 1.0, 10.0]},
    ),
    (
        ElasticNet, "EN",
        {"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.3, 0.7], "max_iter": [5000]},
    ),
    (
        KNeighborsRegressor, "KNN",
        {"n_neighbors": [3, 5, 9], "weights": ["uniform", "distance"]},
    ),
]

if HAS_XGB:
    MODEL_DEFS.append((
        XGBRegressor, "XGB",
        {"n_estimators": [200, 400], "max_depth": [4, 6],
         "learning_rate": [0.05, 0.1], "subsample": [0.8],
         "random_state": [42], "verbosity": [0]},
    ))

if HAS_LGB:
    MODEL_DEFS.append((
        LGBMRegressor, "LGB",
        {"n_estimators": [200, 400], "num_leaves": [31, 63],
         "learning_rate": [0.05, 0.1], "random_state": [42], "verbose": [-1]},
    ))


def get_all_combinations() -> list:
    """
    Expand MODEL_DEFS x FE_GRID into a flat list of run configs.
    Each config: {class, base_name, params (FE + model HPs merged)}.
    """
    combos = []
    fe_keys = list(FE_GRID.keys())
    fe_vals = list(FE_GRID.values())

    for fe_combo in product(*fe_vals):
        fe_params = dict(zip(fe_keys, fe_combo))
        for model_class, base_name, hp_grid in MODEL_DEFS:
            hp_keys = list(hp_grid.keys())
            hp_vals = list(hp_grid.values())
            for hp_combo in product(*hp_vals):
                hp_params = dict(zip(hp_keys, hp_combo))
                combos.append({
                    "class":     model_class,
                    "base_name": base_name,
                    "params":    {**fe_params, **hp_params},
                })
    return combos
