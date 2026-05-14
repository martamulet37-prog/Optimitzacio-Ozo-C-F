"""
04_plots.py
===========
Genera gràfics per als resultats de l'esquema dual.

Per cada target (removed_abs, removed_terb):
  - Pareto scatter (JarTest R2 vs Validació R2)
  - Predicted vs Actual del millor model

Sèrie temporal del millor model ABS + millor model TERB sobreposats.

Execució:
    python 04_plots.py
    python 04_plots.py --n_best 3
"""
import os, sys, pickle, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data_loader import load_real, TARGETS
from src.features    import apply_fe, transform_new

PARETO_PATH = os.path.join(ROOT, "results", "PARETO_SUMMARY.csv")
MODELS_DIR  = os.path.join(ROOT, "results", "models")
IMAGES_DIR  = os.path.join(ROOT, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 9,  "ytick.labelsize": 9, "grid.alpha": 0.4,
})

TARGET_INFO = {
    "removed_abs":  {"tag": "ABS",  "cal_split": "abs_cal",  "val_split": "abs_val"},
    "removed_terb": {"tag": "TERB", "cal_split": "terb_cal", "val_split": "terb_val"},
}


def _save(name):
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(IMAGES_DIR, f"{name}.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Desat: images/{name}.pdf/.png")


# ── Plot 1: Pareto per target ──────────────────────────────────────────────────

def plot_pareto(pareto_df):
    targets = pareto_df["target_label"].unique()
    fig, axes = plt.subplots(1, len(targets), figsize=(6*len(targets), 5))
    if len(targets) == 1:
        axes = [axes]

    for ax, target in zip(axes, targets):
        sub   = pareto_df[pareto_df["target_label"] == target]
        front = sub[sub["pareto_front"]]
        rest  = sub[~sub["pareto_front"]]

        ax.scatter(rest["jt_r2"],  rest["p2_r2"],  c="#aec6cf",
                   edgecolors="grey", lw=0.5, s=35, alpha=0.6, label="No-Pareto")
        ax.scatter(front["jt_r2"], front["p2_r2"], c="#e05252",
                   edgecolors="k", lw=0.8, s=80, zorder=5, label="Front Pareto")

        for _, r in front.nlargest(3, "p2_r2").iterrows():
            short = r["model_id"].split("_")[2]  # base model name
            ax.annotate(short, (r["jt_r2"], r["p2_r2"]),
                        textcoords="offset points", xytext=(5,3),
                        fontsize=8, color="#a00000")

        vals = list(sub["jt_r2"]) + list(sub["p2_r2"])
        lims = [min(vals)-0.05, max(vals)+0.05]
        ax.plot(lims, lims, color="grey", ls="--", lw=1, alpha=0.5, label="y = x")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("JarTest OOF $R^2$")
        ax.set_ylabel("Validació $R^2$")
        cal_label = "P1" if target == "removed_abs" else "P2a"
        val_label = "P2" if target == "removed_abs" else "P2b"
        ax.set_title(f"Pareto — {target}\ncal:{cal_label}  val:{val_label}")
        ax.legend(fontsize=9); ax.grid(True, ls="--")

    plt.tight_layout()
    _save("pareto_analysis")


# ── Plot 2: Predicted vs Actual ────────────────────────────────────────────────

def plot_validation_fit(pareto_df, splits, n_best=3):
    targets = ["removed_abs", "removed_terb"]
    fig, axes = plt.subplots(n_best, len(targets),
                             figsize=(5*len(targets), 4*n_best))
    if n_best == 1:
        axes = axes[np.newaxis, :]

    for col_j, target in enumerate(targets):
        info  = TARGET_INFO[target]
        X_val, y_val_df = splits[info["val_split"]]
        y_val = y_val_df[target].values

        sub_front = pareto_df[(pareto_df["target_label"] == target) &
                               pareto_df["pareto_front"]]
        best_ids  = sub_front.nlargest(n_best, "p2_r2")["model_id"].tolist()

        for row_i in range(n_best):
            ax = axes[row_i, col_j]
            if row_i >= len(best_ids):
                ax.axis("off"); continue

            mid = best_ids[row_i]
            pkl = os.path.join(MODELS_DIR, f"{mid}.pkl")
            if not os.path.exists(pkl):
                ax.axis("off"); continue

            with open(pkl,"rb") as f:
                cd = pickle.load(f)
            Xv_p = transform_new(X_val, cd["params"], cd["transformers"])
            pred = cd["model"].predict(Xv_p)

            r2   = r2_score(y_val, pred)
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            ax.scatter(y_val, pred, c="#008080", edgecolors="k",
                       lw=0.3, s=10, alpha=0.35)
            lims = [min(y_val.min(),pred.min())*0.95,
                    max(y_val.max(),pred.max())*1.05]
            ax.plot(lims, lims, "r--", lw=1)
            ax.set_xlabel(f"Real {target}")
            ax.set_ylabel(f"Predit {target}")
            short = mid.split("_")[2]
            val_label = "P2" if target == "removed_abs" else "P2b"
            ax.set_title(f"{short} | {target} [{val_label}]\n"
                         f"$R^2$={r2:.3f}  RMSE={rmse:.4f}")
            ax.grid(True, ls="--")

    plt.tight_layout()
    _save("validation_fit")


# ── Plot 3: Sèrie temporal ─────────────────────────────────────────────────────

def plot_timeseries(pareto_df, splits):
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    colors = {"cal_real":"#2c7bb6","cal_pred":"#fdae61",
               "val_real":"#1a9641","val_pred":"#d7191c"}

    for ax, target in zip(axes, ["removed_abs","removed_terb"]):
        info = TARGET_INFO[target]
        X_cal, y_cal_df = splits[info["cal_split"]]
        X_val, y_val_df = splits[info["val_split"]]

        sub_front = pareto_df[(pareto_df["target_label"] == target) &
                               pareto_df["pareto_front"]]
        if sub_front.empty:
            ax.set_title(f"{target} — cap model Pareto"); continue

        mid = sub_front.nlargest(1,"p2_r2")["model_id"].iloc[0]
        pkl = os.path.join(MODELS_DIR, f"{mid}.pkl")
        if not os.path.exists(pkl): continue

        with open(pkl,"rb") as f:
            cd = pickle.load(f)
        p, ts = cd["params"], cd["transformers"]

        Xc_p, _, _ = apply_fe(X_cal, X_cal, p)
        pred_cal    = cd["model"].predict(Xc_p)
        pred_val    = cd["model"].predict(transform_new(X_val, p, ts))

        y_cal = y_cal_df[target].values
        y_val = y_val_df[target].values

        ax.plot(y_cal_df.index, y_cal,  color=colors["cal_real"],
                alpha=0.6, lw=0.7, label="Real cal")
        ax.plot(y_cal_df.index, pred_cal, color=colors["cal_pred"],
                alpha=0.6, lw=0.7, label="Predit cal")
        ax.plot(y_val_df.index, y_val,  color=colors["val_real"],
                alpha=0.6, lw=0.7, label="Real val")
        ax.plot(y_val_df.index, pred_val, color=colors["val_pred"],
                alpha=0.6, lw=0.7, label="Predit val")
        ax.axvline(y_val_df.index[0], color="k", ls="--",
                   lw=1.2, label="cal|val")

        cal_label = "P1" if target == "removed_abs" else "P2a"
        val_label = "P2" if target == "removed_abs" else "P2b"
        short = mid.split("_")[2]
        ax.set_title(f"{target}  [{short}]  cal:{cal_label} → val:{val_label}")
        ax.set_ylabel(target)
        ax.legend(fontsize=8, ncol=5)
        ax.grid(True, ls="--")

    plt.tight_layout()
    _save("timeseries_best")


# ── Main ───────────────────────────────────────────────────────────────────────

def run(n_best=3):
    print("\n=== STEP 4: Generació de gràfics ===")
    pareto_df = pd.read_csv(PARETO_PATH)
    splits    = load_real(os.path.join(ROOT, "HISTORIC_COAG.csv"))

    plot_pareto(pareto_df)
    plot_validation_fit(pareto_df, splits, n_best=n_best)
    plot_timeseries(pareto_df, splits)
    print("\n  Tots els gràfics desats a images/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_best", type=int, default=3)
    args = parser.parse_args()
    run(n_best=args.n_best)
