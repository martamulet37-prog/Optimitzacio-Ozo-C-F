# Optimització de Coagulació/Floculació mitjançant ML

## Descripció del projecte

Aquest projecte desenvolupa i valida models de machine learning per a l'optimització del procés de **coagulació i floculació** en una planta potabilitzadora d'aigua. L'objectiu és predir l'eliminació de contaminants (absorbància i terbolesa) en funció dels paràmetres operacionals, a partir de dades de JarTest de laboratori i dades reals de planta.

El sistema segueix un esquema **dual de calibració**: els models entrenen sobre assajos de JarTest controlats i es calibren i validen posteriorment sobre dades reals de planta, separades per règim operacional.

### Variables objectiu (targets)

| Target | Descripció |
|---|---|
| `removed_abs` | Eliminació d'absorbància UV (matèria orgànica) |
| `removed_terb` | Eliminació de terbolesa (NTU) |

### Variables predictores

| Variable | Descripció |
|---|---|
| `COAGULANT` | Dosi de coagulant (mg/L) |
| `FLOCULANT` | Dosi de floculant (mg/L) |
| `O3_dose` | Dosi d'ozó (mg/L) |
| `pH_set` | pH d'operació ajustat |
| `Abs_in` | Absorbància de l'aigua d'entrada |
| `Terb_in` | Terbolesa de l'aigua d'entrada (NTU) |
| `SLUDGE` | Paràmetre de fangs de la planta (constant) |

---

## Estructura del projecte

```
.
├── JARTESTS.xlsx              # Dades d'assajos de JarTest (laboratori)
├── HISTORIC_COAG.csv          # Dades històriques de coagulació (planta real)
├── HISTORIC_PLANTA.csv        # Dades generals de planta
│
├── src/
│   ├── data_loader.py         # Càrrega i divisió de dades (JarTest + planta real)
│   └── features.py            # Enginyeria de features (poly, log1p, escalat, ràtios)
│
├── models/
│   ├── configs.py             # Combinacions d'hiperparàmetres i identificadors
│   └── ensemble.py            # CrossValidationEnsemble (mitja dels folds)
│
├── results/
│   ├── MASTER_LOG.csv         # Resultats benchmark JarTest (tots els models)
│   ├── CALIBRATION_LOG.csv    # Resultats calibració sobre dades reals
│   ├── VALIDATION_LOG.csv     # Resultats validació (conjunt de test)
│   ├── PARETO_SUMMARY.csv     # Front de Pareto (JarTest R² vs Validació R²)
│   └── models/                # Models entrenats serialitzats (.pkl)
│
├── images/                    # Gràfics generats
│
├── 01_jartest_benchmark.py    # Pas 1: benchmark i CV sobre JarTest
├── 02_calibration.py          # Pas 2: calibració dual sobre dades reals
├── 03_validation_pareto.py    # Pas 3: validació i selecció Pareto
├── 04_plots.py                # Pas 4: gràfics de resultats
├── 05_partial_effects.py      # Pas 5: efectes parcials per feature
├── 06_response_surface.py     # Pas 6: superfície de resposta (2D)
│
├── requirements.txt
└── README.md
```

---

## Pipeline d'execució

El pipeline s'executa pas a pas en ordre. Cada script llegeix els resultats dels anteriors des de `results/`.

```bash
# Pas 1 — Benchmark sobre JarTest (10-fold CV, tots els models)
python 01_jartest_benchmark.py

# Pas 2 — Calibració dual dels top-N models sobre dades reals
python 02_calibration.py --top 20

# Pas 3 — Validació i front de Pareto
python 03_validation_pareto.py

# Pas 4 — Gràfics de diagnòstic
python 04_plots.py --n_best 3

# Pas 5 — Efectes parcials per feature
python 05_partial_effects.py --percentile_range 2 98 --n_points 100

# Pas 6 — Superfície de resposta 2D
python 06_response_surface.py
```

---

## Esquema de dades (calibració dual)

| Partició | Descripció | Ús |
|---|---|---|
| **JarTest** | Assajos de laboratori | Entrenament i CV (10-fold) |
| **P1** | Dades reals, règim fix (absorbància) | Calibració `removed_abs` |
| **P2** | Dades reals, règim fix (validació abs) | Validació `removed_abs` |
| **P2a** | Dades reals, règim variable (terbolesa) | Calibració `removed_terb` |
| **P2b** | Dades reals, règim variable (validació terb) | Validació `removed_terb` |

---

## Enginyeria de features (`src/features.py`)

Paràmetres configurables per combinació de model:

| Paràmetre | Descripció |
|---|---|
| `rel_o3` | Ràtios derivades: `O3/coagulant`, `coagulant/floculant` |
| `log1p` | Transformació logarítmica de features no negatives |
| `scaler` | Escalat: `standard` (Z-score), `minmax` o cap |
| `poly` | Expansió polinòmica (grau 1, 2 o 3) |

---

## Selecció de models (front de Pareto)

Els models es seleccionen per **dominància de Pareto** entre dues mètriques:

- **JarTest OOF R²** — robustesa en validació creuada de laboratori
- **Validació R²** — generalització sobre dades reals de planta

Això evita seleccionar models que sobreajusten al JarTest sense generalitzar, o que funcionen bé en planta per casualitat sense base estadística.

---

## Gràfics generats (`images/`)

| Fitxer | Contingut |
|---|---|
| `pareto_analysis.pdf/png` | Scatter Pareto per cada target |
| `validation_fit.pdf/png` | Predit vs Real dels millors models |
| `timeseries_best.pdf/png` | Sèrie temporal (calibració i validació) |
| `partial_effects_removed_abs.pdf/png` | Efectes parcials per `removed_abs` |
| `partial_effects_removed_terb.pdf/png` | Efectes parcials per `removed_terb` |
| `partial_effects_combined.pdf/png` | Variables operacionals vs ambdós targets |

---

## Requisits

Vegeu `requirements.txt` per a la llista completa de dependències.

```bash
pip install -r requirements.txt
```

Python recomanat: **3.10 o superior**.