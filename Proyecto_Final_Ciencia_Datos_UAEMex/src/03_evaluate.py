import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "figure.dpi": 300
})

def main():
    print("1. Evaluando modelos sobre el conjunto de PRUEBA (Test Set)...")
    data = joblib.load("data/processed/dataset_splits.joblib")
    X_test, y_test = data["X_test"], data["y_test"]
    models = joblib.load("results/metrics/trained_models.joblib")
    with open("results/metrics/cv_results.json") as f:
        cv_results = json.load(f)

    results_table = []
    fig_roc, ax_roc = plt.subplots(figsize=(5.5, 4.5))
    fig_pr, ax_pr = plt.subplots(figsize=(5.5, 4.5))
    fig_cm, axes_cm = plt.subplots(2, 2, figsize=(8, 6.8))

    colors = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c"]

    for idx, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)

        results_table.append({
            "Modelo": name,
            "CV F1 (Mean ± Std)": f"{cv_results[name]['cv_mean_f1']:.4f} ± {cv_results[name]['cv_std_f1']:.4f}",
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4),
            "PR-AUC": round(pr_auc, 4)
        })

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", color=colors[idx], lw=1.8)

        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
        ax_pr.plot(recall_vals, precision_vals, label=f"{name} (AP = {pr_auc:.3f})", color=colors[idx], lw=1.8)

        cm = confusion_matrix(y_test, y_pred)
        ax_c = axes_cm[idx // 2, idx % 2]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_c,
                    xticklabels=["Pred: No", "Pred: Sí"], yticklabels=["Real: No", "Real: Sí"])
        ax_c.set_title(f"Matriz de Confusión: {name}")

    ax_roc.plot([0, 1], [0, 1], "k--", lw=1)
    ax_roc.set_xlabel("Tasa de Falsos Positivos (FPR)")
    ax_roc.set_ylabel("Tasa de Verdaderos Positivos (TPR / Recall)")
    ax_roc.set_title("Curvas ROC en Conjunto de Prueba")
    ax_roc.legend(loc="lower right", fontsize=8.5)
    fig_roc.tight_layout()
    fig_roc.savefig("results/figures/fig4_roc_curves.png")
    fig_roc.savefig("paper/figures/fig4_roc_curves.png")

    baseline_pr = y_test.mean()
    ax_pr.axhline(baseline_pr, color="k", linestyle="--", lw=1, label=f"Azar ({baseline_pr:.3f})")
    ax_pr.set_xlabel("Sensibilidad (Recall)")
    ax_pr.set_ylabel("Precisión (Precision)")
    ax_pr.set_title("Curvas Precisión-Recall (Enfoque en Clase Minoritaria)")
    ax_pr.legend(loc="upper right", fontsize=8.5)
    fig_pr.tight_layout()
    fig_pr.savefig("results/figures/fig5_pr_curves.png")
    fig_pr.savefig("paper/figures/fig5_pr_curves.png")

    fig_cm.tight_layout()
    fig_cm.savefig("results/figures/fig6_confusion_matrices.png")
    fig_cm.savefig("paper/figures/fig6_confusion_matrices.png")

    df_results = pd.DataFrame(results_table)
    df_results.to_csv("results/metrics/final_test_comparison.csv", index=False)
    print("\n=== TABLA COMPARATIVA FINAL (CONJUNTO DE PRUEBA) ===")
    print(df_results.to_string(index=False))

    mlp_model = models["MLP_NeuralNet"]
    y_pred_mlp = mlp_model.predict(X_test)
    fp = np.sum((y_pred_mlp == 1) & (y_test == 0))
    fn = np.sum((y_pred_mlp == 0) & (y_test == 1))
    print(f"\nAnálisis de errores (MLP): Falsos Positivos = {fp}, Falsos Negativos = {fn}")

if __name__ == "__main__":
    main()
