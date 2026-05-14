"""
data_loader.py
==============
Carrega dades de JarTest i de planta real.

Targets (ELIMINACIÓ):
    removed_abs  = Abs_in  - Abs_out
    removed_terb = Terb_in - Terb_out

Esquema de splits
-----------------
removed_abs  : calibra P1  (jul23–set24)  →  valida P2  (oct24–nov25)
removed_terb : calibra P2a (oct24–abr25)  →  valida P2b (mai25–nov25)

Raó: a P1 Terb_in és sempre 39.3 NTU (sensor saturat / règim fix).
     El model no pot aprendre variabilitat que no ha vist.
     P2a cobreix el règim real de turbidesa variable.
"""
import pandas as pd
import numpy as np

FEATURES     = ["Abs_in", "Terb_in", "O3_dose", "pH_set", "COAGULANT", "FLOCULANT", "SLUDGE"]
TARGETS      = ["removed_abs", "removed_terb"]
PLANT_SLUDGE = 1

# ── Dates de tall ─────────────────────────────────────────────────────────────
SPLIT_ABS  = "2024-09-28"   # P1 / P2 per removed_abs
SPLIT_TERB = "2025-04-30"   # P2a / P2b per removed_terb (meitat de P2)

# ── Llindars outliers ─────────────────────────────────────────────────────────
OUTLIER_FILTERS = {
    "Terb_in_max":     45.0,
    "Terb_out_max":     5.0,
    "removed_abs_min": -1.0,
    "removed_abs_max":  7.0,
}


def _add_removals(df):
    df = df.copy()
    df["removed_abs"]  = df["Abs_in"]  - df["Abs_out"]
    df["removed_terb"] = df["Terb_in"] - df["Terb_out"]
    return df


def _apply_outlier_filters(df, verbose=True):
    n0 = len(df)
    f  = OUTLIER_FILTERS
    mask = (
        (df["Terb_in"]     <= f["Terb_in_max"])     &
        (df["Terb_out"]    <= f["Terb_out_max"])    &
        (df["removed_abs"] >= f["removed_abs_min"]) &
        (df["removed_abs"] <= f["removed_abs_max"])
    )
    df_net = df[mask].copy()
    if verbose:
        n_eli = n0 - len(df_net)
        print(f"[outliers] Eliminades {n_eli:,} files ({100*n_eli/n0:.1f}%)"
              f"  →  {len(df_net):,} files netes")
    return df_net


def load_jartest(path="JARTESTS.xlsx"):
    """Retorna (X, y) del JarTest."""
    df = pd.read_excel(path)
    df = _add_removals(df)
    df = df.dropna(subset=FEATURES + TARGETS)
    return (df[FEATURES].copy().reset_index(drop=True),
            df[TARGETS].copy().reset_index(drop=True))


def load_real(path="HISTORIC_COAG.csv", filter_outliers=True):
    """
    Carrega dades reals i retorna els 4 splits amb esquema dual:

        X_abs_cal,  y_abs_cal   → calibrar   removed_abs  (P1)
        X_abs_val,  y_abs_val   → validar     removed_abs  (P2 sencer)
        X_terb_cal, y_terb_cal  → calibrar   removed_terb (P2a)
        X_terb_val, y_terb_val  → validar     removed_terb (P2b)

    Retorna dict amb claus: abs_cal, abs_val, terb_cal, terb_val
    Cada valor és un tuple (X, y).
    """
    df = pd.read_csv(path)
    df["TS"] = pd.to_datetime(df["TS"])
    df = df.set_index("TS").sort_index()
    df["SLUDGE"] = PLANT_SLUDGE
    df = _add_removals(df)

    if filter_outliers:
        df = _apply_outlier_filters(df)

    df = df.dropna(subset=FEATURES + TARGETS)

    # ── Splits ────────────────────────────────────────────────────────────────
    p1  = df[df.index <= SPLIT_ABS]
    p2  = df[df.index >  SPLIT_ABS]
    p2a = p2[p2.index <= SPLIT_TERB]
    p2b = p2[p2.index >  SPLIT_TERB]

    def XY(subset):
        return subset[FEATURES].copy(), subset[TARGETS].copy()

    splits = {
        "abs_cal":  XY(p1),    # removed_abs  — calibració
        "abs_val":  XY(p2),    # removed_abs  — validació
        "terb_cal": XY(p2a),   # removed_terb — calibració
        "terb_val": XY(p2b),   # removed_terb — validació
    }

    print(f"[data] abs  calibració (P1)  : {p1.index.min().date()} -> "
          f"{p1.index.max().date()}  ({len(p1):,} files)")
    print(f"[data] abs  validació  (P2)  : {p2.index.min().date()} -> "
          f"{p2.index.max().date()}  ({len(p2):,} files)")
    print(f"[data] terb calibració (P2a) : {p2a.index.min().date()} -> "
          f"{p2a.index.max().date()}  ({len(p2a):,} files)")
    print(f"[data] terb validació  (P2b) : {p2b.index.min().date()} -> "
          f"{p2b.index.max().date()}  ({len(p2b):,} files)")

    return splits
