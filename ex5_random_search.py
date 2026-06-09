import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common_tp import (
    FIG_DIR,
    best_so_far,
    prepare_data,
    print_device,
    random_search,
    set_seed,
)


search_space = {
    "lr": ("log_uniform", 1e-4, 1e-2),
    "weight_decay": ("log_uniform", 1e-5, 1e-2),
    "dropout_rate": ("uniform", 0.0, 0.5),
    "clip_value": ("uniform", 0.5, 5.0),
    "hidden_dims": (
        "choice",
        [[64, 32], [128, 64], [128, 64, 32], [256, 128, 64], [256, 128, 64, 32]],
    ),
    "activation": ("choice", ["relu", "leaky_relu", "elu", "selu"]),
}


def main():
    set_seed()
    print_device()
    train_loader, val_loader, _, _, _, _ = prepare_data(scale=True)
    rs_results = random_search(search_space, 48, train_loader, val_loader, epochs=80)
    rs_results.to_csv("results/random_search_results.csv", index=False)

    print("\n=== TOP 10 configurations Random Search ===")
    cols = ["hidden_dims", "activation", "dropout_rate", "lr", "weight_decay", "clip_value", "val_mse"]
    print(rs_results[cols].head(10).to_string(index=False))

    gs_path = "results/grid_search_results.csv"
    try:
        gs_results = pd.read_csv(gs_path)
        plt.figure(figsize=(9, 5))
        plt.plot(best_so_far(gs_results), label="Grid Search")
        plt.plot(best_so_far(rs_results), label="Random Search")
        plt.xlabel("Nombre de configurations evaluees")
        plt.ylabel("Meilleur val_MSE trouve")
        plt.title("Convergence : Grid Search vs Random Search")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "ex5_gs_vs_rs.png", dpi=120)
        plt.close()
    except FileNotFoundError:
        print("grid_search_results.csv absent : lancez ex4_grid_search.py pour la comparaison.")

    plt.figure(figsize=(8, 5))
    sc = plt.scatter(
        rs_results["lr"],
        rs_results["val_mse"],
        c=rs_results["dropout_rate"],
        cmap="viridis",
    )
    plt.xscale("log")
    plt.xlabel("lr")
    plt.ylabel("val_mse")
    plt.colorbar(sc, label="dropout_rate")
    plt.title("Random Search : lr vs val_mse")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ex5_lr_scatter.png", dpi=120)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=rs_results, x="weight_decay", y="val_mse", hue="activation")
    plt.xscale("log")
    plt.title("Random Search : weight_decay vs val_mse")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ex5_weight_decay_scatter.png", dpi=120)
    plt.close()

    tmp = pd.concat(
        [
            rs_results.head(10).assign(groupe="top10"),
            rs_results.tail(10).assign(groupe="pire10"),
        ]
    )
    plt.figure(figsize=(7, 5))
    sns.violinplot(data=tmp, x="groupe", y="dropout_rate")
    plt.title("Dropout : top10 vs pire10")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ex5_dropout_violin.png", dpi=120)
    plt.close()

    continuous = ["lr", "weight_decay", "dropout_rate", "clip_value"]
    corr_rows = []
    for hp in continuous:
        corr = rs_results[[hp, "val_mse"]].corr(method="spearman").iloc[0, 1]
        corr_rows.append({"hyperparametre": hp, "spearman_val_mse": corr})
    corr_df = pd.DataFrame(corr_rows).sort_values("spearman_val_mse", key=lambda s: s.abs(), ascending=False)
    corr_df.to_csv("results/ex5_spearman_correlations.csv", index=False)
    print("\nCorrelations Spearman :")
    print(corr_df.to_string(index=False))


if __name__ == "__main__":
    main()
