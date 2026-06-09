import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import shapiro
from torch.utils.data import DataLoader, TensorDataset

from common_tp import (
    DeepFFN,
    FIG_DIR,
    MODEL_DIR,
    DEVICE,
    RESULTS_DIR,
    evaluate,
    load_dataframe,
    model_kwargs,
    predict,
    prepare_data,
    print_device,
    set_seed,
    train_model,
)


baseline_config = {
    "hidden_dims": [128, 64, 32],
    "activation": "relu",
    "use_bn": True,
    "dropout_rate": 0.2,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "clip_value": 1.0,
}


def read_results(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()


def parse_hidden_dims(value):
    if isinstance(value, list):
        return value
    return [int(x.strip()) for x in str(value).strip("[]").split(",") if x.strip()]


def row_to_config(row):
    return {
        "hidden_dims": parse_hidden_dims(row["hidden_dims"]),
        "activation": row["activation"],
        "use_bn": bool(row.get("use_bn", True)),
        "dropout_rate": float(row["dropout_rate"]),
        "lr": float(row["lr"]),
        "weight_decay": float(row["weight_decay"]),
        "clip_value": float(row["clip_value"]),
    }


def build_top_configs():
    gs = read_results(RESULTS_DIR / "grid_search_results.csv")
    rs = read_results(RESULTS_DIR / "random_search_results.csv")
    configs = {"Baseline": {**baseline_config, "_source_val_mse": np.nan}}
    candidates = []

    if not gs.empty:
        best = gs.sort_values("val_mse").iloc[0]
        cfg = {**row_to_config(best), "_source_val_mse": float(best["val_mse"])}
        configs["Meilleur Grid Search"] = cfg
        candidates.append(("Meilleur Grid Search", best["val_mse"], cfg))

    if not rs.empty:
        for i, (_, row) in enumerate(rs.sort_values("val_mse").head(3).iterrows(), start=1):
            name = f"Random Search #{i}"
            cfg = {**row_to_config(row), "_source_val_mse": float(row["val_mse"])}
            configs[name] = cfg
            candidates.append((name, row["val_mse"], cfg))

    if not candidates:
        print("Resultats grid/random absents : utilisation de variantes baseline pour ex6.")
        configs["Variante baseline large"] = {
            **baseline_config,
            "hidden_dims": [256, 128, 64],
            "_source_val_mse": np.nan,
        }
        configs["Variante baseline faible dropout"] = {
            **baseline_config,
            "dropout_rate": 0.1,
            "_source_val_mse": np.nan,
        }

    return configs


def make_report(final_df, shapiro_stat, shapiro_p):
    pdf_path = "rapport_synthese.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.axis("off")
        ax.set_title("Tableau de synthese comparatif", fontsize=16)
        table = ax.table(
            cellText=final_df.round(4).values,
            colLabels=final_df.columns,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 7))
        ax.axis("off")
        conclusion = (
            "Conclusion\n\n"
            "Les experiences montrent que la normalisation des donnees et BatchNorm stabilisent fortement "
            "l'optimisation du MLP. Dropout et L2 jouent surtout un role de regularisation lorsque l'ecart "
            "train/validation augmente. Le gradient clipping protege contre les pics de gradient, en particulier "
            "sur les donnees brutes non standardisees. Grid Search est simple et exhaustif sur un espace discret, "
            "mais son cout augmente tres vite. Random Search explore mieux les valeurs continues comme lr et "
            "weight_decay avec le meme budget. Les limites principales sont le cout de calcul, la sensibilite au "
            "split, et le fait que le dataset possede une cible plafonnee a 5.0. Des perspectives naturelles sont "
            "Bayesian Optimization, Hyperband, une validation croisee plus complete et des modeles ensembles.\n\n"
            f"Test Shapiro-Wilk sur residus : statistique={shapiro_stat:.4f}, p-value={shapiro_p:.4g}."
        )
        ax.text(0.02, 0.95, conclusion, va="top", wrap=True, fontsize=12)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    print(f"Rapport genere : {pdf_path}")


def main():
    set_seed()
    print_device()
    train_loader, val_loader, test_loader, _, tensors, _ = prepare_data(scale=True)
    configs = build_top_configs()

    X_trainval = torch.cat([tensors["X_train_t"], tensors["X_val_t"]], dim=0)
    y_trainval = torch.cat([tensors["y_train_t"], tensors["y_val_t"]], dim=0)
    trainval_loader = DataLoader(
        TensorDataset(X_trainval, y_trainval),
        batch_size=64,
        shuffle=True,
        pin_memory=(DEVICE.type == "cuda"),
    )

    final_results = []
    best_model, best_mse, best_name = None, float("inf"), None
    for name, config in configs.items():
        set_seed()
        model = DeepFFN(**model_kwargs(config))
        source_val_mse = config.get("_source_val_mse", np.nan)
        config_full = {
            **{k: v for k, v in config.items() if not k.startswith("_")},
            "epochs": 300,
            "early_stopping_patience": 30,
        }
        _, _, elapsed = train_model(
            model,
            trainval_loader,
            trainval_loader,
            config_full,
            save_path=MODEL_DIR / f"final_{name.replace(' ', '_').replace('#', '')}.pth",
            verbose=False,
        )
        model.load_state_dict(torch.load(MODEL_DIR / f"final_{name.replace(' ', '_').replace('#', '')}.pth", map_location=DEVICE))
        model = model.to(DEVICE)
        te_mse, te_mae, te_r2 = evaluate(model, test_loader, nn.MSELoss())
        final_results.append(
            {
                "Modele/Config": name,
                "Val MSE": source_val_mse,
                "Test MSE": te_mse,
                "Test MAE": te_mae,
                "Test R2": te_r2,
                "Temps total": elapsed,
            }
        )
        print(f"{name} -> test_MSE={te_mse:.4f} MAE={te_mae:.4f} R2={te_r2:.4f}")
        if te_mse < best_mse:
            best_mse, best_model, best_name = te_mse, model, name

    final_df = pd.DataFrame(final_results)
    final_df.to_csv(RESULTS_DIR / "final_test_results.csv", index=False)

    y_true, y_pred = predict(best_model, test_loader)
    residuals = y_pred - y_true
    ci95 = 1.96 * residuals.std() / np.sqrt(len(residuals))

    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, s=10, alpha=0.35)
    low, high = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([low, high], [low, high], color="red", label="y=x")
    plt.fill_between([low, high], [low - ci95, high - ci95], [low + ci95, high + ci95], alpha=0.15)
    plt.xlabel("Valeurs reelles")
    plt.ylabel("Predictions")
    plt.title(f"Predit vs Reel - {best_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ex6_pred_vs_real.png", dpi=120)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=50, kde=True)
    plt.axvline(0, color="red", linestyle="--")
    plt.title("Distribution des residus")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ex6_residuals_distribution.png", dpi=120)
    plt.close()

    shapiro_sample = residuals[: min(5000, len(residuals))]
    shapiro_stat, shapiro_p = shapiro(shapiro_sample)
    print(f"Shapiro-Wilk residus : stat={shapiro_stat:.4f}, p={shapiro_p:.4g}")

    data, df = load_dataframe()
    _, _, _, _, _, raw = prepare_data(scale=False)
    X_test_raw = raw["X_test_raw"]
    lon_idx = data.feature_names.index("Longitude")
    lat_idx = data.feature_names.index("Latitude")
    plt.figure(figsize=(8, 7))
    sc = plt.scatter(
        X_test_raw[:, lon_idx],
        X_test_raw[:, lat_idx],
        c=residuals,
        cmap="coolwarm",
        s=12,
        alpha=0.65,
    )
    plt.colorbar(sc, label="Erreur signee (prediction - reel)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Erreurs par region")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ex6_regional_errors.png", dpi=120)
    plt.close()

    make_report(final_df, shapiro_stat, shapiro_p)


if __name__ == "__main__":
    main()
