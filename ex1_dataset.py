import matplotlib.pyplot as plt
import seaborn as sns

from common_tp import FIG_DIR, load_dataframe, prepare_data, print_device, set_seed


def main():
    set_seed()
    print_device()

    data, df = load_dataframe()
    print(f"Nombre d'exemples : {df.shape[0]}")
    print(f"Nombre de features : {len(data.feature_names)}")
    print(f"Noms des features : {data.feature_names}")
    print("\nCinq premieres lignes :")
    print(df.head())
    print("\nStatistiques descriptives :")
    print(df.describe())

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(df["MedHouseVal"], bins=50, kde=True)
    plt.title("Distribution de MedHouseVal")
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df["MedHouseVal"])
    plt.title("Boxplot de MedHouseVal")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ex1_target_distribution.png", dpi=120)
    plt.close()

    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Heatmap de correlation")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ex1_correlation_heatmap.png", dpi=120)
    plt.close()

    target_corr = corr["MedHouseVal"].drop("MedHouseVal").abs().sort_values(ascending=False)
    feature_corr = corr.drop(index="MedHouseVal", columns="MedHouseVal").abs()
    for col in feature_corr.columns:
        feature_corr.loc[col, col] = 0
    pair = feature_corr.stack().sort_values(ascending=False).index[0]

    print("\nCommentaire Q2 :")
    print("La cible est asymetrique a droite et plafonnee a 5.0, ce qui cree un effet de censure.")
    print("Le boxplot montre des valeurs elevees proches du plafond artificiel.")
    print("\nCommentaire Q3 :")
    print(f"Feature la plus correlee a la cible : {target_corr.index[0]} ({target_corr.iloc[0]:.3f})")
    print(f"Paire de features la plus colineaire : {pair[0]} / {pair[1]}")

    train_loader, val_loader, test_loader, _, _, _ = prepare_data(scale=True)
    xb, yb = next(iter(train_loader))
    print("\nVerification DataLoader :")
    print(f"X batch : {xb.shape} | y batch : {yb.shape}")
    print(f"X mean : {xb.mean():.4f} | X std : {xb.std():.4f}")
    print(f"Val batch size : {next(iter(val_loader))[0].shape}")
    print(f"Test batch size : {next(iter(test_loader))[0].shape}")
    print("\nCommentaire Q5 :")
    print("Le scaler est fitte seulement sur train pour eviter le data leakage vers validation/test.")


if __name__ == "__main__":
    main()
