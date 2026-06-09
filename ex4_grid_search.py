import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common_tp import FIG_DIR, grid_search, prepare_data, print_device, set_seed


param_grid = {
    "hidden_dims": [[64, 32], [128, 64], [128, 64, 32], [256, 128, 64]],
    "activation": ["relu", "leaky_relu", "elu"],
    "dropout_rate": [0.0, 0.2, 0.3],
    "lr": [1e-3, 5e-4],
    "weight_decay": [0.0, 1e-4],
    "clip_value": [1.0, 5.0],
}

param_grid_small = {
    "hidden_dims": [[64, 32], [128, 64, 32], [256, 128, 64]],
    "activation": ["relu", "leaky_relu"],
    "dropout_rate": [0.1, 0.3],
    "lr": [1e-3, 5e-4],
    "weight_decay": [1e-4, 1e-3],
    "clip_value": [1.0],
}


def architecture_size(hidden_dims):
    return sum(hidden_dims)


def main():
    set_seed()
    print_device()
    n_configs = 1
    for values in param_grid.values():
        n_configs *= len(values)
    print(f"Total configurations Grid Search : {n_configs}")

    train_loader, val_loader, _, _, _, _ = prepare_data(scale=True)
    gs_results = grid_search(param_grid_small, train_loader, val_loader, epochs=80)
    gs_results.to_csv("results/grid_search_results.csv", index=False)

    print("\n=== TOP 10 configurations Grid Search ===")
    cols = ["hidden_dims", "activation", "dropout_rate", "lr", "weight_decay", "val_mse"]
    print(gs_results[cols].head(10).to_string(index=False))

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=gs_results, x="activation", y="val_mse")
    plt.title("Grid Search : activation")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ex4_boxplot_activation.png", dpi=120)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=gs_results, x="dropout_rate", y="val_mse")
    plt.title("Grid Search : dropout_rate")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ex4_boxplot_dropout.png", dpi=120)
    plt.close()

    pivot = gs_results.pivot_table(values="val_mse", index="lr", columns="weight_decay", aggfunc="mean")
    plt.figure(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="viridis")
    plt.title("MSE moyen : lr x weight_decay")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ex4_heatmap_lr_wd.png", dpi=120)
    plt.close()

    top15 = gs_results.head(15).copy()
    top15["arch_size"] = top15["hidden_dims"].apply(architecture_size)
    top15["config"] = [f"#{i + 1}" for i in range(len(top15))]
    plt.figure(figsize=(10, 5))
    sns.barplot(data=top15, x="config", y="val_mse", hue="arch_size", palette="mako")
    plt.title("Top 15 Grid Search")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ex4_top15_barplot.png", dpi=120)
    plt.close()

    rows = []
    for hp in ["hidden_dims", "activation", "dropout_rate", "lr", "weight_decay", "clip_value"]:
        std_mean = gs_results.groupby(gs_results[hp].astype(str))["val_mse"].mean().std()
        rows.append({"hyperparametre": hp, "std_des_moyennes_val_mse": std_mean})
    impact = pd.DataFrame(rows).sort_values("std_des_moyennes_val_mse", ascending=False)
    impact.to_csv("results/ex4_hyperparam_impact.csv", index=False)
    print("\nImpact individuel estime :")
    print(impact.to_string(index=False))


if __name__ == "__main__":
    main()
