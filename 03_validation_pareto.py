"""
03_validation_pareto.py
=======================
Valida els models calibrats sobre el conjunt de validació corresponent:

    CALIB_ABS_*  → valida sobre P2  (removed_abs)
    CALIB_TERB_* → valida sobre P2b (removed_terb)

Pareto: JarTest R2 vs Validació R2, per cada target i per la mitjana.

Execució:
    python 03_validation_pareto.py
"""
import os, sys, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data_loader import load_real, TARGETS
from src.features    import transform_new

CALIB_PATH  = os.path.join(ROOT, "results", "CALIBRATION_LOG.csv")
LOG_PATH    = os.path.join(ROOT, "results", "MASTER_LOG.csv")
VAL_PATH    = os.path.join(ROOT, "results", "VALIDATION_LOG.csv")
PARETO_PATH = os.path.join(ROOT, "results", "PARETO_SUMMARY.csv")
MODELS_DIR  = os.path.join(ROOT, "results", "models")


def _r2_rmse(yt, yp, prefix):
    return {
        f"{prefix}r2":   float(r2_score(yt, yp)),
        f"{prefix}rmse": float(np.sqrt(mean_squared_error(yt, yp))),
    }


def _dominated(jt_r2, val_r2):
    n   = len(jt_r2)
    dom = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and jt_r2[j] >= jt_r2[i] and val_r2[j] >= val_r2[i] \
               and (jt_r2[j] > jt_r2[i] or val_r2[j] > val_r2[i]):
                dom[i] = True; break
    return dom


def run():
    print("\n=== STEP 3: Validació + Pareto (esquema dual) ===")
    splits = load_real(os.path.join(ROOT, "HISTORIC_COAG.csv"))
    X_abs_val,  y_abs_val  = splits["abs_val"]
    X_terb_val, y_terb_val = splits["terb_val"]

    calib_log = pd.read_csv(CALIB_PATH)
    jt_log    = pd.read_csv(LOG_PATH)[["model_id","jt_r2_removed_abs",
                                        "jt_r2_removed_terb","jt_r2_mean"]].copy()

    rows = []
    for _, row in calib_log.iterrows():
        calib_id = row["model_id"]
        pkl = os.path.join(MODELS_DIR, f"{calib_id}.pkl")
        if not os.path.exists(pkl): continue

        with open(pkl, "rb") as f:
            cd = pickle.load(f)

        target = cd["target"]
        if target == "removed_abs":
            X_val = X_abs_val
            y_val = y_abs_val["removed_abs"].values
        else:
            X_val = X_terb_val
            y_val = y_terb_val["removed_terb"].values

        Xv_p = transform_new(X_val, cd["params"], cd["transformers"])
        pred = cd["model"].predict(Xv_p)
        m    = _r2_rmse(y_val, pred, prefix=f"p2_")
        rows.append({**row.to_dict(), "target": target, **m})

    val_df = pd.DataFrame(rows)
    val_df.to_csv(VAL_PATH, index=False)
    print(f"  Validats {len(rows)} models.")

    # ── Pareto per cada target ─────────────────────────────────────────────────
    pareto_frames = []
    for target, jt_col in [("removed_abs",  "jt_r2_removed_abs"),
                            ("removed_terb", "jt_r2_removed_terb")]:
        sub = val_df[val_df["target"] == target].copy()
        if sub.empty: continue

        base_id = sub["model_id"].str.replace(f"CALIB_{'ABS' if 'abs' in target else 'TERB'}_", "")
        jt_sub  = jt_log.set_index("model_id").reindex(base_id.values)
        sub["jt_r2"] = jt_sub[jt_col].values

        dom = _dominated(sub["jt_r2"].values, sub["p2_r2"].values)
        sub["pareto_front"] = ~dom
        sub["target_label"] = target
        pareto_frames.append(sub)

    pareto_df = pd.concat(pareto_frames).sort_values(
        ["target_label","pareto_front","p2_r2"],
        ascending=[True, False, False]).reset_index(drop=True)
    pareto_df.to_csv(PARETO_PATH, index=False)

    for target in ["removed_abs", "removed_terb"]:
        sub   = pareto_df[pareto_df["target_label"] == target]
        front = sub[sub["pareto_front"]]
        if front.empty: continue
        best  = front.iloc[0]
        print(f"\n  [{target}]  front Pareto: {len(front)} models")
        print(f"    Millor: {best['model_id']}")
        print(f"    JarTest R2  = {best['jt_r2']:.4f}")
        print(f"    Validació R2 = {best['p2_r2']:.4f}   RMSE = {best['p2_rmse']:.4f}")

    return pareto_df


if __name__ == "__main__":
    run()
