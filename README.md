# TP2 — Deep Feedforward Networks for Regression

A six-part lab study of a **deep feedforward network** on the California Housing dataset, in
PyTorch. It runs the full experimental arc: exploratory analysis, a baseline model, a regularisation
ablation, grid search, random search, and a final held-out evaluation with residual diagnostics.

Each exercise is a standalone script writing figures to `figures/` and captured output to
`results/`.

> Code comments, printed output and the synthesis report are in **French**; this README is in
> English.

---

## Contents

- [Dataset](#dataset)
- [Model](#model)
- [Exercises](#exercises)
- [Results](#results)
- [Running](#running)
- [Repository layout](#repository-layout)

---

## Dataset

**California Housing** (`sklearn.datasets.fetch_california_housing`) — 20,640 districts,
8 numeric features, regression target `MedHouseVal` (median house value).

| Feature | Meaning |
|---|---|
| `MedInc` | Median income |
| `HouseAge` | Median house age |
| `AveRooms` | Average rooms per household |
| `AveBedrms` | Average bedrooms per household |
| `Population` | Block population |
| `AveOccup` | Average occupancy |
| `Latitude` / `Longitude` | Location |

One property of this dataset shapes every result that follows: **the target is capped at 5.0**.
`MedHouseVal` is right-skewed with a hard ceiling, so the top of the range is censored rather than
observed — visible as a spike at 5.0 in `figures/ex1_target_distribution.png`. No model can do
better than the ceiling allows, and residuals are structurally non-normal as a result.

Features are standardised (`StandardScaler`) and split into train/validation/test by
`prepare_data()` in `common_tp.py`.

## Model

`DeepFFN` in [`common_tp.py`](common_tp.py) — a configurable feedforward network.

```
Input(8)
  [ Linear -> BatchNorm -> Activation -> Dropout ] x len(hidden_dims)
  Linear(-> 1)
```

Defaults: `hidden_dims=[128, 64, 32]`, `activation='relu'`, `use_bn=True`, `dropout_rate=0.2`.

Activations available: `relu`, `tanh`, `leaky_relu`, `elu`, `selu`.

Weight initialisation is chosen to match the activation — **He/Kaiming normal** for ReLU-family
activations (`relu`, `leaky_relu`, `elu`) and **Xavier uniform** otherwise, with biases zeroed.
That pairing is the textbook-correct choice and is implemented explicitly in `_init_weights()`
rather than left to PyTorch's defaults.

Training (`train_model`) uses gradient clipping (`clip_value=1.0` by default), early stopping on
validation MSE, and records a history for the learning curves.

## Exercises

| Script | Subject | Output |
|---|---|---|
| `ex1_dataset.py` | Exploratory analysis — distributions, correlations | `outputex1.md`, 2 figures |
| `ex2_model.py` | Baseline model and learning curves | `outputex2.md`, 1 figure |
| `ex3_ablation.py` | Regularisation ablation + gradient clipping | `ex3_ablation_results.csv`, 5 figures |
| `ex4_grid_search.py` | Exhaustive grid search | `grid_search_results.csv`, `ex4_hyperparam_impact.csv`, 4 figures |
| `ex5_random_search.py` | Random search + Spearman analysis | `random_search_results.csv`, `ex5_spearman_correlations.csv`, 4 figures |
| `ex6_evaluation.py` | Final test evaluation + residual diagnostics | `final_test_results.csv`, 3 figures, `rapport_synthese.pdf` |

## Results

### Final test-set comparison

From [`results/final_test_results.csv`](results/final_test_results.csv):

| Configuration | Val MSE | Test MSE | Test MAE | Test R² | Time (s) |
|---|---|---|---|---|---|
| Baseline | — | 0.2949 | 0.3796 | 0.7772 | 98.6 |
| Best Grid Search | 0.2883 | 0.2771 | 0.3665 | 0.7907 | 277.0 |
| **Random Search #1** | **0.2779** | **0.2509** | **0.3377** | **0.8105** | 244.7 |
| Random Search #2 | 0.2861 | 0.3217 | 0.3908 | 0.7570 | 83.5 |
| Random Search #3 | 0.2913 | 0.2841 | 0.3594 | 0.7854 | 204.4 |

**Random search beat grid search** at comparable cost — 0.2509 vs 0.2771 test MSE, R² 0.811 vs
0.791. That is the expected outcome when only a few hyperparameters matter: random search samples
more distinct values along the important axes, while grid search spends its budget on a dense
lattice that repeats the same values of the influential parameter.

Note also that **Random Search #2 has a better validation MSE than #3 but a worse test MSE**
(0.2861/0.3217 vs 0.2913/0.2841). With this validation-set size, differences of ~0.005 in validation
MSE do not reliably predict test ranking — a caution against reading too much into a leaderboard.

### Ablation — which regularisation actually helps

From [`results/ex3_ablation_results.csv`](results/ex3_ablation_results.csv), best validation MSE
(lower is better):

| Configuration | Best val MSE | Time (s) |
|---|---|---|
| A — baseline (BN + Dropout + L2) | 0.3796 | 29.9 |
| **B — no BatchNorm** | **0.2596** | 97.3 |
| C — no Dropout | 0.3297 | 33.6 |
| D — no L2 | 0.3555 | 50.3 |
| E — no regularisation at all | 0.2687 | 48.1 |

The result is counter-intuitive and worth stating plainly: **removing BatchNorm gave the best
validation MSE**, and removing *all* regularisation came second. The fully regularised baseline was
the worst of the five.

The likely explanation is that this problem is too small and too easy to need this much
regularisation — 8 features, 20,640 rows, a smooth target. BatchNorm plus 0.2 dropout plus L2 is
enough combined constraint to underfit. Note too that config B took 3× longer to converge (97 s vs
30 s), consistent with BatchNorm's usual role of speeding up convergence even where it does not
improve the final result.

### What the hyperparameters actually do

Grid search variance ([`results/ex4_hyperparam_impact.csv`](results/ex4_hyperparam_impact.csv)) —
standard deviation of mean validation MSE per hyperparameter:

| Hyperparameter | Std of mean val MSE |
|---|---|
| `hidden_dims` | 0.0324 |
| `dropout_rate` | 0.0207 |
| `lr` | 0.0022 |
| `weight_decay` | 0.0012 |
| `activation` | 0.0009 |

Random search Spearman correlation with validation MSE
([`results/ex5_spearman_correlations.csv`](results/ex5_spearman_correlations.csv)):

| Hyperparameter | Spearman ρ |
|---|---|
| `lr` | **−0.307** |
| `clip_value` | −0.091 |
| `dropout_rate` | −0.015 |
| `weight_decay` | −0.010 |

The two methods agree on the conclusion while measuring different things: **architecture
(`hidden_dims`) and `dropout_rate` dominate the grid's variance, and learning rate is the only
parameter with a meaningful monotonic relationship to the score.** Activation choice is nearly
irrelevant (σ = 0.0009), and `weight_decay` barely registers on either measure.

The negative ρ for `lr` means higher learning rates in the sampled range gave *lower* (better) MSE.

### Residual diagnostics

`ex6_evaluation.py` runs a Shapiro-Wilk test on the residuals:

```
stat = 0.8877,  p = 8.829e-43
```

Normality is **decisively rejected**. This is the expected consequence of the target's cap at 5.0 —
censored targets produce a pile-up of residuals at the ceiling. It is a property of the data, not a
defect in the model, and it means prediction intervals assuming Gaussian errors would be wrong.

Figures: `ex6_pred_vs_real.png`, `ex6_residuals_distribution.png`, `ex6_regional_errors.png` (errors
by geography).

## Running

Requires Python 3.10+, PyTorch, scikit-learn, pandas, numpy, matplotlib, seaborn.

```bash
pip install torch scikit-learn pandas numpy matplotlib seaborn

python ex1_dataset.py
python ex2_model.py
python ex3_ablation.py
python ex4_grid_search.py
python ex5_random_search.py
python ex6_evaluation.py
```

Run them in order — `ex6` consumes the models and results produced by the earlier scripts.

`common_tp.py` creates `figures/`, `models/` and `results/` on import and selects CUDA when
available (the committed runs used an NVIDIA GTX 1650 Ti; on CPU expect the searches to take
considerably longer). `matplotlib` is set to the `Agg` backend, so everything runs headless.

Seed is fixed at `SEED = 42` via `set_seed()`.

> There is no `requirements.txt`. The dataset downloads automatically on first run via
> `fetch_california_housing`.

## Repository layout

```
common_tp.py        Shared library - data, DeepFFN, training loop, grid/random search, plotting
ex1_dataset.py      Exploratory analysis
ex2_model.py        Baseline
ex3_ablation.py     Regularisation ablation + gradient clipping
ex4_grid_search.py  Grid search
ex5_random_search.py Random search + Spearman
ex6_evaluation.py   Final evaluation + report generation
figures/            19 generated figures
results/            CSVs + captured console output per exercise
rapport_synthese.pdf  Synthesis report (French), generated by ex6
```

### Shared helpers in `common_tp.py`

| Function | Role |
|---|---|
| `set_seed`, `print_device` | Reproducibility and device reporting |
| `load_dataframe`, `prepare_data` | Loading, scaling, DataLoaders |
| `DeepFFN`, `model_kwargs` | Model and config mapping |
| `train_one_epoch`, `evaluate`, `predict`, `train_model` | Training loop with clipping and early stopping |
| `grid_search`, `sample_config`, `random_search`, `best_so_far` | Hyperparameter search |
| `plot_history` | Learning curves |
