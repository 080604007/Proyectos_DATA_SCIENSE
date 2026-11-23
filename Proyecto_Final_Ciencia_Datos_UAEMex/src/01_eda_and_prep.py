import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 300
})

def load_dataset():
    raw_path = "data/raw/bank_marketing_raw.csv"
    try:
        from ucimlrepo import fetch_ucirepo
        print("1. Intentando descargar dataset oficial Bank Marketing (UCI ID: 222)...")
        bank_marketing = fetch_ucirepo(id=222)
        X = bank_marketing.data.features.copy()
        y = bank_marketing.data.targets.copy()
        df = pd.concat([X, y], axis=1)
        df.to_csv(raw_path, index=False)
        print("   Descarga oficial desde UCI exitosa.")
        return df
    except Exception as e:
        print(f"   Aviso: usando copia local pre-cargada en {raw_path} ({e})")
        return pd.read_csv(raw_path)

def main():
    df = load_dataset()
    print(f"   Dataset cargado: {df.shape[0]} registros, {df.shape[1]} columnas.")

    y_bin = (df["y"] == "yes").astype(int)
    X_features = df.drop(columns=["y"])

    # Prevención de Data Leakage: eliminar 'duration' según recomendación oficial de UCI
    if "duration" in X_features.columns:
        print("   Eliminando variable 'duration' para evitar fuga de datos (data leakage).")
        X_features = X_features.drop(columns=["duration"])

    print("2. Generando gráficas de Análisis Exploratorio...")
    
    # Gráfica 1: Proporción de clases
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    counts = y_bin.value_counts()
    bars = ax.bar(["No Suscribe (0)", "Suscribe (1)"], counts.values, color=["#4C72B0", "#DD8452"], edgecolor="black")
    ax.set_title("Distribución de la Variable Objetivo (Desbalance de Clases)")
    ax.set_ylabel("Número de Clientes")
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:,} ({h/len(y_bin)*100:.1f}%)",
                    xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(0, max(counts.values) * 1.15)
    plt.tight_layout()
    plt.savefig("results/figures/fig1_class_distribution.png")
    plt.savefig("paper/figures/fig1_class_distribution.png")
    plt.close()

    # Gráfica 2: Distribuciones numéricas clave
    num_cols_plot = ["age", "balance", "campaign", "pdays"]
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    for idx, col in enumerate(num_cols_plot):
        ax = axes[idx // 2, idx % 2]
        sns.boxplot(data=df, x="y", y=col, hue="y", legend=False, ax=ax, showfliers=False, palette=["#4C72B0", "#DD8452"])
        ax.set_title(f"Distribución de '{col}' por Clase")
        ax.set_xlabel("Suscripción (y)")
        ax.set_ylabel(col)
    plt.tight_layout()
    plt.savefig("results/figures/fig2_numerical_distributions.png")
    plt.savefig("paper/figures/fig2_numerical_distributions.png")
    plt.close()

    # Gráfica 3: Matriz de correlación
    num_cols = X_features.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X_features.select_dtypes(include=["object", "category"]).columns.tolist()

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    corr = X_features[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Matriz de Correlación (Variables Numéricas)")
    plt.tight_layout()
    plt.savefig("results/figures/fig3_correlation_matrix.png")
    plt.savefig("paper/figures/fig3_correlation_matrix.png")
    plt.close()

    # Partición estratificada 70% / 15% / 15%
    print("3. Dividiendo en Entrenamiento (70%), Validación (15%) y Prueba (15%)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_features, y_bin, test_size=0.30, stratify=y_bin, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols)
        ]
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    joblib.dump({
        "X_train": X_train_proc, "y_train": y_train.values,
        "X_val": X_val_proc, "y_val": y_val.values,
        "X_test": X_test_proc, "y_test": y_test.values,
        "feature_names": preprocessor.get_feature_names_out().tolist()
    }, "data/processed/dataset_splits.joblib")

    eda_summary = {
        "total_samples": int(len(df)),
        "missing_values_per_col": df.isnull().sum().to_dict(),
        "class_0_count": int(counts[0]),
        "class_1_count": int(counts[1]),
        "num_features": num_cols,
        "cat_features": cat_cols,
        "processed_dim": int(X_train_proc.shape[1])
    }
    with open("results/metrics/eda_summary.json", "w") as f:
        json.dump(eda_summary, f, indent=4)

    print(f"Preprocesamiento finalizado. Dimensión final de características: {X_train_proc.shape[1]}")

if __name__ == "__main__":
    main()
