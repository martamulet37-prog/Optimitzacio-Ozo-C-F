"""
01_jartest_benchmark.py
=======================
Entrena tots els models sobre dades de JarTest (10-fold CV).
No canvia respecte versions anteriors — els JarTests cobreixen tots dos targets.

Execució:
    python 01_jartest_benchmark.py
"""
import os, sys, pickle, datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data_loader  import load_jartest, TARGETS
from src.features     import apply_fe
from models.configs   import get_combos, model_hp, build_id
from models.ensemble  import CrossValidationEnsemble

LOG_PATH   = os.path.join(ROOT, "results", "MASTER_LOG.csv")
MODELS_DIR = os.path.join(ROOT, "results", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def metrics(y_true, y_pred, prefix=""):
    out = {}
    for j, t in enumerate(TARGETS):
        yt, yp = y_true[:, j], y_pred[:, j]
        out[f"{prefix}r2_{t}"]   = float(r2_score(yt, yp))
        out[f"{prefix}rmse_{t}"] = float(np.sqrt(mean_squared_error(yt, yp)))
    out[f"{prefix}r2_mean"]   = float(np.mean([out[f"{prefix}r2_{t}"]   for t in TARGETS]))
    out[f"{prefix}rmse_mean"] = float(np.mean([out[f"{prefix}rmse_{t}"] for t in TARGETS]))
    return out


def run():
    print("\n=== STEP 1: JarTest benchmark (10-fold CV) ===")
    X, y = load_jartest(os.path.join(ROOT, "JARTESTS.xlsx"))
    y_arr   = y.values
    combos  = get_combos()
    kf      = KFold(n_splits=10, shuffle=True, random_state=33)

    existing = set(pd.read_csv(LOG_PATH)["model_id"].tolist()) \
               if os.path.exists(LOG_PATH) else set()

    n_new = 0
    for i, combo in enumerate(combos):
        p   = combo["params"]
        mid = build_id(combo["base_name"], p)
        if mid in existing:
            continue
        print(f"  [{i+1}/{len(combos)}] {mid}")

        oof = np.zeros_like(y_arr, dtype=float)
        fold_models, fold_t = [], []

        for tr_idx, vl_idx in kf.split(X):
            Xtr, Xvl = X.iloc[tr_idx], X.iloc[vl_idx]
            ytr       = y_arr[tr_idx]
            Xtr_p, Xvl_p, t_set = apply_fe(Xtr, Xvl, p)
            base  = combo["class"](**model_hp(p))
            model = MultiOutputRegressor(base, n_jobs=1)
            model.fit(Xtr_p, ytr)
            fold_models.append(model)
            fold_t.append(t_set)
            oof[vl_idx] = model.predict(Xvl_p)

        ens = CrossValidationEnsemble(fold_models, p, fold_t)
        with open(os.path.join(MODELS_DIR, f"{mid}_ensemble.pkl"), "wb") as f:
            pickle.dump(ens, f)

        m = metrics(y_arr, oof, prefix="jt_")
        entry = {"timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 "model_id": mid, "params": str(p), **m}
        header = not os.path.exists(LOG_PATH)
        pd.DataFrame([entry]).to_csv(LOG_PATH, mode="a", index=False, header=header)
        existing.add(mid)
        n_new += 1

    log = pd.read_csv(LOG_PATH).sort_values("jt_r2_mean", ascending=False)
    print(f"\n  {n_new} models nous entrenats. Total: {len(log)}")
    best = log.iloc[0]
    print(f"  Millor JarTest : {best['model_id']}")
    print(f"    jt_r2_removed_abs  = {best['jt_r2_removed_abs']:.4f}")
    print(f"    jt_r2_removed_terb = {best['jt_r2_removed_terb']:.4f}")
    return log


if __name__ == "__main__":
    run()
