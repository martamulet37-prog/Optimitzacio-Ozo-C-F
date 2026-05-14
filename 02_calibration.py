"""
02_calibration.py
=================
Calibra els top-N models JarTest sobre dades reals amb esquema dual:

    removed_abs  → calibra sobre P1  (règim conegut)
    removed_terb → calibra sobre P2a (règim real de turbidesa variable)

Desa un model calibrat per target: CALIB_ABS_<id>.pkl  i  CALIB_TERB_<id>.pkl

Execució:
    python 02_calibration.py
    python 02_calibration.py --top 30
"""
import os, sys, pickle, argparse
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data_loader import load_real, TARGETS
from src.features    import apply_fe
from models.configs  import model_hp
from models.ensemble import CrossValidationEnsemble

LOG_PATH   = os.path.join(ROOT, "results", "MASTER_LOG.csv")
CALIB_PATH = os.path.join(ROOT, "results", "CALIBRATION_LOG.csv")
MODELS_DIR = os.path.join(ROOT, "results", "models")


def _metrics_single(y_true_col, y_pred_col, name):
    r2   = float(r2_score(y_true_col, y_pred_col))
    rmse = float(np.sqrt(mean_squared_error(y_true_col, y_pred_col)))
    return {f"r2_{name}": r2, f"rmse_{name}": rmse}


def _calibrate_one_target(mid, ens, X_cal, y_cal_col, tag, existing):
    """
    Reentrena l'estimador sobre X_cal/y_cal_col (un sol target).
    tag: 'ABS' o 'TERB'
    Retorna (calib_id, model, t_set, metrics_dict) o None si ja existeix.
    """
    calib_id = f"CALIB_{tag}_{mid}"
    if calib_id in existing:
        return None

    p = ens.params
    X_cal_p, _, t_set = apply_fe(X_cal, X_cal, p)

    # Regressió simple (un sol target)
    base_cls = type(ens.fold_models[0].estimators_[0])
    from sklearn.base import clone
    model = base_cls(**model_hp(p))
    model.fit(X_cal_p, y_cal_col)

    pred = model.predict(X_cal_p)
    name = "removed_abs" if tag == "ABS" else "removed_terb"
    m = _metrics_single(y_cal_col, pred, f"cal_{name}")

    return calib_id, model, t_set, p, m


def run(top_n=20):
    print(f"\n=== STEP 2: Calibració dual (top {top_n} models) ===")
    splits = load_real(os.path.join(ROOT, "HISTORIC_COAG.csv"))

    X_abs_cal,  y_abs_cal  = splits["abs_cal"]
    X_terb_cal, y_terb_cal = splits["terb_cal"]

    jt_log   = pd.read_csv(LOG_PATH).sort_values("jt_r2_mean", ascending=False)
    top_rows = jt_log.head(top_n)

    existing = set(pd.read_csv(CALIB_PATH)["model_id"].tolist()) \
               if os.path.exists(CALIB_PATH) else set()

    n_new = 0
    for _, row in top_rows.iterrows():
        mid = row["model_id"]
        pkl = os.path.join(MODELS_DIR, f"{mid}_ensemble.pkl")
        if not os.path.exists(pkl):
            print(f"  [SKIP] {mid} — ensemble no trobat"); continue

        with open(pkl, "rb") as f:
            ens: CrossValidationEnsemble = pickle.load(f)

        for tag, X_cal, y_df, target in [
            ("ABS",  X_abs_cal,  y_abs_cal,  "removed_abs"),
            ("TERB", X_terb_cal, y_terb_cal, "removed_terb"),
        ]:
            result = _calibrate_one_target(
                mid, ens, X_cal, y_df[target].values, tag, existing)
            if result is None:
                continue
            calib_id, model, t_set, p, m = result

            with open(os.path.join(MODELS_DIR, f"{calib_id}.pkl"), "wb") as f:
                pickle.dump({"model": model, "params": p,
                             "transformers": t_set, "target": target}, f)

            entry = {"model_id": calib_id, "target": target,
                     "base_jt_r2_mean": row["jt_r2_mean"], **m}
            header = not os.path.exists(CALIB_PATH)
            pd.DataFrame([entry]).to_csv(
                CALIB_PATH, mode="a", index=False, header=header)
            existing.add(calib_id)
            n_new += 1
            print(f"  {calib_id[:60]}  r2={list(m.values())[0]:.4f}")

    print(f"\n  {n_new} models nous calibrats ({n_new//2} per target).")
    return pd.read_csv(CALIB_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    run(top_n=args.top)
