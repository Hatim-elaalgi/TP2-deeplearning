import itertools
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
for directory in (FIG_DIR, MODEL_DIR, RESULTS_DIR):
    directory.mkdir(exist_ok=True)

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == "cuda"

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_device():
    print(f"Device utilise : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU : {torch.cuda.get_device_name(0)}")


def load_dataframe():
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["MedHouseVal"] = data.target
    return data, df


def prepare_data(scale=True, batch_size=64, val_test_batch_size=256):
    data, df = load_dataframe()
    X = df[data.feature_names].values.astype(np.float32)
    y = df["MedHouseVal"].values.astype(np.float32).reshape(-1, 1)

    bins = pd.cut(df["MedHouseVal"], bins=10, labels=False, duplicates="drop")
    X_train, X_tmp, y_train, y_tmp, bins_train, bins_tmp = train_test_split(
        X, y, bins, test_size=0.30, random_state=SEED, stratify=bins
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=bins_tmp
    )

    scaler = StandardScaler()
    if scale:
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    else:
        scaler.fit(X_train)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_ds, batch_size=val_test_batch_size, shuffle=False, pin_memory=PIN_MEMORY
    )
    test_loader = DataLoader(
        test_ds, batch_size=val_test_batch_size, shuffle=False, pin_memory=PIN_MEMORY
    )

    tensors = {
        "X_train_t": X_train_t,
        "y_train_t": y_train_t,
        "X_val_t": X_val_t,
        "y_val_t": y_val_t,
        "X_test_t": X_test_t,
        "y_test_t": y_test_t,
    }
    raw = {
        "df": df,
        "data": data,
        "train_index_like": None,
        "X_test_raw": X_test,
        "y_test": y_test,
    }
    return train_loader, val_loader, test_loader, scaler, tensors, raw


class DeepFFN(nn.Module):
    """
    Reseau feedforward profond avec architecture configurable,
    BatchNorm, Dropout et initialisation He/Xavier.
    """

    def __init__(
        self,
        input_dim=8,
        hidden_dims=None,
        output_dim=1,
        activation="relu",
        use_bn=True,
        dropout_rate=0.2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        self.activation_name = activation
        self.use_bn = use_bn
        self.dropout_rate = dropout_rate
        self.layers = nn.ModuleList()

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            block = nn.ModuleDict()
            block["linear"] = nn.Linear(prev_dim, hidden_dim)
            if use_bn:
                block["bn"] = nn.BatchNorm1d(hidden_dim)
            block["activation"] = self._get_activation(activation)
            if dropout_rate > 0:
                block["dropout"] = nn.Dropout(dropout_rate)
            self.layers.append(block)
            prev_dim = hidden_dim

        self.output_layer = nn.Linear(prev_dim, output_dim)
        self._init_weights()

    def _get_activation(self, name):
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(negative_slope=0.01),
            "elu": nn.ELU(),
            "selu": nn.SELU(),
        }
        if name not in activations:
            raise ValueError(f"Activation inconnue : {name}")
        return activations[name]

    def _init_weights(self):
        he_activations = {"relu", "leaky_relu", "elu"}
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.activation_name in he_activations:
                    nonlinearity = "leaky_relu" if self.activation_name == "leaky_relu" else "relu"
                    nn.init.kaiming_normal_(module.weight, nonlinearity=nonlinearity)
                else:
                    nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        for block in self.layers:
            x = block["linear"](x)
            if "bn" in block:
                x = block["bn"](x)
            x = block["activation"](x)
            if "dropout" in block:
                x = block["dropout"](x)
        return self.output_layer(x)


def model_kwargs(config):
    keys = DeepFFN.__init__.__code__.co_varnames
    return {k: v for k, v in config.items() if k in keys}


def train_one_epoch(model, loader, optimizer, criterion, clip_value=1.0):
    model.train()
    total_loss, total_gnorm, n = 0.0, 0.0, 0
    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()

        if clip_value is None:
            total_norm = torch.sqrt(
                sum(
                    (p.grad.detach().norm(2) ** 2)
                    for p in model.parameters()
                    if p.grad is not None
                )
            )
        else:
            total_norm = nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step()
        total_loss += loss.item() * len(xb)
        total_gnorm += float(total_norm)
        n += len(xb)

    return total_loss / n, total_gnorm / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    y_true, y_pred = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        pred = model(xb).detach().cpu().numpy()
        y_pred.append(pred)
        y_true.append(yb.numpy())

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, r2


@torch.no_grad()
def predict(model, loader):
    model.eval()
    y_true, y_pred = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        y_pred.append(model(xb).detach().cpu().numpy())
        y_true.append(yb.numpy())
    return np.vstack(y_true).ravel(), np.vstack(y_pred).ravel()


def train_model(model, train_loader, val_loader, config, save_path=None, verbose=True):
    model = model.to(DEVICE)
    save_path = Path(save_path or MODEL_DIR / "best_model.pth")
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 0.0),
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.5, min_lr=1e-6
    )
    criterion = nn.MSELoss()
    clip_value = config.get("clip_value", 1.0)

    history = {
        "train_mse": [],
        "val_mse": [],
        "val_mae": [],
        "val_r2": [],
        "lr": [],
        "grad_norm": [],
    }
    best_val = float("inf")
    patience = config.get("early_stopping_patience", 20)
    no_improve = 0
    t0 = time.time()

    for epoch in range(config.get("epochs", 200)):
        tr_loss, gnorm = train_one_epoch(
            model, train_loader, optimizer, criterion, clip_value
        )
        va_mse, va_mae, va_r2 = evaluate(model, val_loader, criterion)
        scheduler.step(va_mse)

        history["train_mse"].append(tr_loss)
        history["val_mse"].append(va_mse)
        history["val_mae"].append(va_mae)
        history["val_r2"].append(va_r2)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["grad_norm"].append(gnorm)

        if va_mse < best_val:
            best_val = va_mse
            no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  Early stopping a l'epoch {epoch + 1}")
                break

        if verbose and (epoch + 1) % 20 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Ep {epoch + 1:3d} | tr_MSE={tr_loss:.4f} | "
                f"val_MSE={va_mse:.4f} | R2={va_r2:.4f} | lr={lr:.6f}"
            )

    return history, best_val, time.time() - t0


def plot_history(history, path, title="Courbes d'apprentissage"):
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.ravel()
    curves = [
        ("train_mse", "Train MSE"),
        ("val_mse", "Validation MSE"),
        ("val_r2", "Validation R2"),
        ("lr", "Learning rate"),
        ("grad_norm", "Norme du gradient"),
    ]
    for ax, (key, label) in zip(axes, curves):
        ax.plot(history[key])
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    axes[-1].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def grid_search(param_grid, train_loader, val_loader, epochs=80):
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))
    results = []

    print(f"Grid Search : {len(combos)} configurations x {epochs} epochs max")
    print("-" * 60)

    for i, combo in enumerate(combos):
        config = dict(zip(keys, combo))
        config["epochs"] = epochs
        config["early_stopping_patience"] = 15
        config["use_bn"] = True

        set_seed(SEED)
        model = DeepFFN(**model_kwargs(config))
        _, best_mse, elapsed = train_model(
            model,
            train_loader,
            val_loader,
            config,
            save_path=MODEL_DIR / f"grid_{i + 1}.pth",
            verbose=False,
        )
        results.append({**config, "val_mse": best_mse, "time_s": elapsed, "order": i + 1})
        print(f"  [{i + 1:3d}/{len(combos)}] val_MSE={best_mse:.4f} ({elapsed:.1f}s)")

    return pd.DataFrame(results).sort_values("val_mse")


def sample_config(space):
    config = {}
    for key, spec in space.items():
        dist = spec[0]
        if dist == "log_uniform":
            value = float(np.exp(np.random.uniform(np.log(spec[1]), np.log(spec[2]))))
        elif dist == "uniform":
            value = float(random.uniform(spec[1], spec[2]))
        elif dist == "choice":
            value = random.choice(spec[1])
        else:
            raise ValueError(f"Distribution inconnue : {dist}")
        config[key] = value
    return config


def random_search(search_space, n_trials, train_loader, val_loader, epochs=80):
    results = []
    print(f"Random Search : {n_trials} tirages x {epochs} epochs max")
    print("-" * 60)

    for trial in range(n_trials):
        config = sample_config(search_space)
        config["epochs"] = epochs
        config["early_stopping_patience"] = 15
        config["use_bn"] = True

        set_seed(trial)
        model = DeepFFN(**model_kwargs(config))
        _, best_mse, elapsed = train_model(
            model,
            train_loader,
            val_loader,
            config,
            save_path=MODEL_DIR / f"random_{trial + 1}.pth",
            verbose=False,
        )

        results.append({**config, "val_mse": best_mse, "trial": trial, "time_s": elapsed, "order": trial + 1})
        print(f"  Trial {trial + 1:3d}/{n_trials} | val_MSE={best_mse:.4f} ({elapsed:.1f}s)")
        print(
            f"  lr={config['lr']:.2e} wd={config['weight_decay']:.2e} "
            f"dr={config['dropout_rate']:.2f} act={config['activation']}"
        )

    return pd.DataFrame(results).sort_values("val_mse")


def best_so_far(df_results):
    df_ordered = df_results.sort_values("order") if "order" in df_results else df_results
    return np.minimum.accumulate(df_ordered["val_mse"].values)
