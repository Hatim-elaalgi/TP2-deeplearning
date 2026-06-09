import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from common_tp import (
    DeepFFN,
    FIG_DIR,
    MODEL_DIR,
    model_kwargs,
    prepare_data,
    print_device,
    set_seed,
    train_model,
)


base = {
    "hidden_dims": [128, 64, 32],
    "activation": "relu",
    "lr": 1e-3,
    "clip_value": 1.0,
    "epochs": 100,
    "early_stopping_patience": 15,
}


def run_config(name, config, train_loader, val_loader, scale_label="scaled"):
    set_seed()
    model = DeepFFN(**model_kwargs(config))
    history, best_mse, elapsed = train_model(
        model,
        train_loader,
        val_loader,
        config,
        save_path=MODEL_DIR / f"ex3_{scale_label}_{name}.pth",
        verbose=False,
    )
    print(f"{name:18s} | val_MSE={best_mse:.4f} | temps={elapsed:.1f}s")
    return history, best_mse, elapsed


def main():
    set_seed()
    print_device()
    train_loader, val_loader, _, _, _, _ = prepare_data(scale=True)

    configs = {
        "A_baseline": {**base, "use_bn": True, "dropout_rate": 0.2, "weight_decay": 1e-4},
        "B_sans_bn": {**base, "use_bn": False, "dropout_rate": 0.2, "weight_decay": 1e-4},
        "C_sans_dropout": {**base, "use_bn": True, "dropout_rate": 0.0, "weight_decay": 1e-4},
        "D_sans_l2": {**base, "use_bn": True, "dropout_rate": 0.2, "weight_decay": 0.0},
        "E_aucune_reg": {**base, "use_bn": False, "dropout_rate": 0.0, "weight_decay": 0.0},
    }

    histories, rows = {}, []
    for name, cfg in configs.items():
        hist, mse, elapsed = run_config(name, cfg, train_loader, val_loader)
        histories[name] = hist
        rows.append({"configuration": name, "best_val_mse": mse, "time_s": elapsed})

    pd.DataFrame(rows).to_csv("results/ex3_ablation_results.csv", index=False)

    for other in ["B_sans_bn", "C_sans_dropout", "E_aucune_reg"]:
        plt.figure(figsize=(8, 5))
        plt.plot(histories["A_baseline"]["val_mse"], label="A_baseline")
        plt.plot(histories[other]["val_mse"], label=other)
        plt.xlabel("Epoch")
        plt.ylabel("Validation MSE")
        plt.title(f"Ablation : A_baseline vs {other}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"ex3_ablation_A_vs_{other}.png", dpi=120)
        plt.close()

    clip_values = [None, 0.1, 0.5, 1.0, 5.0, 10.0]
    clip_histories = {}
    for clip in clip_values:
        label = "sans_clipping" if clip is None else f"clip_{clip}"
        cfg = {**configs["A_baseline"], "clip_value": clip, "epochs": 50}
        hist, _, _ = run_config(label, cfg, train_loader, val_loader)
        clip_histories[label] = hist

    plt.figure(figsize=(10, 6))
    for label, hist in clip_histories.items():
        plt.plot(hist["grad_norm"], label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Norme gradient")
    plt.title("Effet du gradient clipping")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ex3_clip_grad_norm_curves.png", dpi=120)
    plt.close()

    grad_rows = []
    for label, hist in clip_histories.items():
        for value in hist["grad_norm"]:
            grad_rows.append({"clip_value": label, "grad_norm": value})
    grad_df = pd.DataFrame(grad_rows)
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=grad_df, x="clip_value", y="grad_norm")
    plt.xticks(rotation=30)
    plt.title("Distribution des normes de gradient")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ex3_clip_grad_norm_boxplot.png", dpi=120)
    plt.close()

    raw_train, raw_val, _, _, _, _ = prepare_data(scale=False)
    for clip in [None, 1.0]:
        label = "raw_sans_clipping" if clip is None else "raw_clip_1"
        cfg = {**configs["A_baseline"], "clip_value": clip, "epochs": 50}
        hist, _, _ = run_config(label, cfg, raw_train, raw_val, scale_label="raw")
        plt.plot(hist["val_mse"], label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Validation MSE")
    plt.title("Donnees brutes : avec vs sans clipping")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ex3_raw_data_clipping_comparison.png", dpi=120)
    plt.close()

    print("\nCommentaire : BatchNorm stabilise souvent le plus l'apprentissage.")
    print("Dropout et L2 reduisent le surapprentissage, surtout lorsque les courbes train/val divergent.")
    print("Un clip_value autour de 1 ou 5 est generalement un bon compromis sur donnees standardisees.")


if __name__ == "__main__":
    main()
