"""
05_partial_effects.py
=====================
Partial Effects Plots (PEP) per als millors models calibrats.

Per cada feature operacional, fixa totes les altres al seu valor
típic (mediana) i varia la feature d'interès al llarg del seu rang
observat. Mostra la resposta predita del model.

Execució:
    python 05_partial_effects.py
    python 05_partial_effects.py --percentile_range 5 95   (rang més conservador)
    python 05_partial_effects.py --n_points 200            (més resolució)
"""
import os, sys, pickle, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data_loader import load_real, FEATURES, PLANT_SLUDGE
from src.features    import transform_new

PARETO_PATH = os.path.join(ROOT, "results", "PARETO_SUMMARY.csv")
MODELS_DIR  = os.path.join(ROOT, "results", "models")
IMAGES_DIR  = os.path.join(ROOT, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 9,  "ytick.labelsize": 9, "grid.alpha": 0.35,
})

# ── Features operacionals (les que l'operari pot canviar) ─────────────────────
OPERATIONAL = ["O3_dose", "pH_set", "COAGULANT", "FLOCULANT"]
INPUT_VARS  = ["Abs_in", "Terb_in"]
ALL_VARY    = OPERATIONAL + INPUT_VARS

FEATURE_LABELS = {
    "O3_dose":   "Dosi O₃ (mg/L)",
    "pH_set":    "pH ajustat",
    "COAGULANT": "Coagulant (mg/L)",
    "FLOCULANT": "Floculant (mg/L)",
    "Abs_in":    "Absorbància entrada",
    "Terb_in":   "Turbidesa entrada (NTU)",
}

TARGET_LABELS = {
    "removed_abs":  "Eliminació absorbància",
    "removed_terb": "Eliminació turbidesa (NTU)",
}


def _load_best_model(target):
    """Carrega el millor model Pareto per al target donat."""
    pareto = pd.read_csv(PARETO_PATH)
    tag    = "ABS" if target == "removed_abs" else "TERB"

    sub = pareto[pareto["target_label"] == target]
    front = sub[sub["pareto_front"]]
    if front.empty:
        front = sub  # fallback: millor disponible

    best_id = front.nlargest(1, "p2_r2")["model_id"].iloc[0]
    pkl     = os.path.join(MODELS_DIR, f"{best_id}.pkl")

    with open(pkl, "rb") as f:
        cd = pickle.load(f)
    print(f"  [{target}] model: {best_id}  (R²={front.iloc[0]['p2_r2']:.3f})")
    return cd


def _build_baseline(X_ref: pd.DataFrame) -> pd.Series:
    """Baseline = mediana de cada feature sobre les dades de referència."""
    baseline = X_ref[FEATURES].median()
    baseline["SLUDGE"] = PLANT_SLUDGE
    return baseline


def _partial_effect(cd, baseline: pd.Series, feature: str,
                    feat_values: np.ndarray) -> np.ndarray:
    """
    Fixa totes les features a baseline, varia 'feature' al llarg de feat_values.
    Retorna les prediccions del model per cada valor.
    """
    rows = []
    for v in feat_values:
        row = baseline.copy()
        row[feature] = v
        rows.append(row)
    X_sweep = pd.DataFrame(rows, columns=FEATURES)
    X_t     = transform_new(X_sweep, cd["params"], cd["transformers"])
    return cd["model"].predict(X_t)


def _feature_range(X_ref: pd.DataFrame, feature: str,
                   plo: float, phi: float, n: int) -> np.ndarray:
    """Rang de la feature entre percentils plo i phi."""
    lo = np.percentile(X_ref[feature].dropna(), plo)
    hi = np.percentile(X_ref[feature].dropna(), phi)
    return np.linspace(lo, hi, n)


def plot_partial_effects(plo=2, phi=98, n_points=100):
    print("\n=== STEP 5: Partial Effects Plots ===")

    # Carrega dades de referència (P2 per defecte — règim operacional actual)
    splits   = load_real(os.path.join(ROOT, "HISTORIC_COAG.csv"))
    X_ref_abs, _  = splits["abs_val"]
    X_ref_terb, _ = splits["terb_val"]

    targets_info = [
        ("removed_abs",  X_ref_abs),
        ("removed_terb", X_ref_terb),
    ]

    n_feat = len(ALL_VARY)

    for target, X_ref in targets_info:
        print(f"\n  Target: {target}")
        cd       = _load_best_model(target)
        baseline = _build_baseline(X_ref)

        fig = plt.figure(figsize=(18, 10))
        fig.suptitle(
            f"Partial Effects — {TARGET_LABELS[target]}\n"
            f"(totes les altres features fixades a la mediana)",
            fontsize=14, fontweight="bold", y=1.01,
        )
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

        axes = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(n_feat)]

        for ax, feat in zip(axes, ALL_VARY):
            feat_vals = _feature_range(X_ref, feat, plo, phi, n_points)
            preds     = _partial_effect(cd, baseline, feat, feat_vals)

            # Línia principal
            ax.plot(feat_vals, preds, color="#008080", lw=2.2, zorder=3)

            # Banda de confiança empírica: repetim el sweep amb ±1 std de cada
            # altra feature per mostrar sensibilitat al context
            baseline_lo = baseline.copy()
            baseline_hi = baseline.copy()
            for other in FEATURES:
                if other != feat and other != "SLUDGE":
                    s = X_ref[other].std()
                    baseline_lo[other] = baseline[other] - 0.5 * s
                    baseline_hi[other] = baseline[other] + 0.5 * s

            preds_lo = _partial_effect(cd, baseline_lo, feat, feat_vals)
            preds_hi = _partial_effect(cd, baseline_hi, feat, feat_vals)
            ax.fill_between(feat_vals,
                            np.minimum(preds_lo, preds_hi),
                            np.maximum(preds_lo, preds_hi),
                            color="#008080", alpha=0.15, zorder=2,
                            label="±0.5 std context")

            # Línia vertical al valor de la baseline (mediana)
            ax.axvline(baseline[feat], color="#e05252", lw=1.2,
                       linestyle="--", label=f"mediana ({baseline[feat]:.2f})")

            ax.set_xlabel(FEATURE_LABELS.get(feat, feat))
            ax.set_ylabel(TARGET_LABELS[target])
            ax.set_title(FEATURE_LABELS.get(feat, feat))
            ax.legend(fontsize=7, loc="best")
            ax.grid(True, linestyle="--")

        plt.tight_layout()
        fname = f"partial_effects_{target}"
        for ext in ("pdf", "png"):
            plt.savefig(os.path.join(IMAGES_DIR, f"{fname}.{ext}"),
                        dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Desat: images/{fname}.pdf/.png")

    # ── Plot combinat: operacionals vs els dos targets ─────────────────────────
    print("\n  Generant plot combinat (operacionals, ambdós targets)...")
    fig, axes = plt.subplots(2, len(OPERATIONAL),
                             figsize=(5 * len(OPERATIONAL), 8),
                             sharey="row")
    fig.suptitle(
        "Partial Effects — Variables operacionals\n"
        "(inputs fixats a la mediana del règim actual)",
        fontsize=13, fontweight="bold",
    )

    for row_i, (target, X_ref) in enumerate(targets_info):
        cd       = _load_best_model(target)
        baseline = _build_baseline(X_ref)

        for col_j, feat in enumerate(OPERATIONAL):
            ax        = axes[row_i, col_j]
            feat_vals = _feature_range(X_ref, feat, plo, phi, n_points)
            preds     = _partial_effect(cd, baseline, feat, feat_vals)

            ax.plot(feat_vals, preds, color="#2c7bb6" if row_i == 0 else "#1a9641",
                    lw=2.2)
            ax.axvline(baseline[feat], color="#e05252", lw=1.2, ls="--",
                       alpha=0.8)
            ax.set_xlabel(FEATURE_LABELS.get(feat, feat))
            if col_j == 0:
                ax.set_ylabel(TARGET_LABELS[target], fontsize=10)
            if row_i == 0:
                ax.set_title(FEATURE_LABELS.get(feat, feat))
            ax.grid(True, linestyle="--")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(IMAGES_DIR, f"partial_effects_combined.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close()
    print("  Desat: images/partial_effects_combined.pdf/.png")
    print("\n  Tots els partial effects desats.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--percentile_range", nargs=2, type=float,
                        default=[2, 98], metavar=("LO", "HI"),
                        help="Percentils de rang per a cada feature (default: 2 98)")
    parser.add_argument("--n_points", type=int, default=100,
                        help="Punts per a cada sweep (default: 100)")
    args = parser.parse_args()
    plot_partial_effects(
        plo=args.percentile_range[0],
        phi=args.percentile_range[1],
        n_points=args.n_points,
    )
