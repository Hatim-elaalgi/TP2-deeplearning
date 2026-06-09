import torch
import torch.nn as nn

from common_tp import (
    DeepFFN,
    FIG_DIR,
    MODEL_DIR,
    DEVICE,
    model_kwargs,
    plot_history,
    prepare_data,
    print_device,
    set_seed,
    train_model,
)


config_baseline = {
    "hidden_dims": [128, 64, 32],
    "activation": "relu",
    "use_bn": True,
    "dropout_rate": 0.2,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "clip_value": 1.0,
    "epochs": 200,
    "early_stopping_patience": 25,
}


def main():
    set_seed()
    print_device()
    train_loader, val_loader, _, _, _, _ = prepare_data(scale=True)

    model = DeepFFN()
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametres entrainables : {total_params:,}")

    xb, _ = next(iter(train_loader))
    out = model(xb)
    print(f"Sortie : {out.shape}")
    print(f"Plage : [{out.min().item():.3f}, {out.max().item():.3f}]")

    model = DeepFFN(**model_kwargs(config_baseline))
    history, best_val_mse, elapsed = train_model(
        model,
        train_loader,
        val_loader,
        config_baseline,
        save_path=MODEL_DIR / "baseline_best.pth",
    )
    plot_history(history, FIG_DIR / "ex2_baseline_learning_curves.png")
    print(f"Baseline best val MSE : {best_val_mse:.4f}")
    print(f"Temps entrainement : {elapsed:.1f} s")
    print(f"Modele sauvegarde : {MODEL_DIR / 'baseline_best.pth'}")


if __name__ == "__main__":
    main()
