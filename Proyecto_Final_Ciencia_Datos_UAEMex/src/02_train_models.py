import json
import time
import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def main():
    print("1. Cargando datos preprocesados...")
    data = joblib.load("data/processed/dataset_splits.joblib")
    X_train, y_train = data["X_train"], data["y_train"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring_metric = "f1"

    # Submuestra estratificada de 6,000 registros para evitar cuello de botella O(n^3) en SVM
    sss = StratifiedShuffleSplit(n_splits=1, train_size=6000, random_state=RANDOM_STATE)
    sub_idx, _ = next(sss.split(X_train, y_train))
    X_train_svm, y_train_svm = X_train[sub_idx], y_train[sub_idx]

    models_config = {
        "Baseline_LogReg": {
            "estimator": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
            "params": {"C": [0.01, 0.1, 1.0, 10.0], "penalty": ["l2"]},
            "search_type": "grid",
            "X": X_train, "y": y_train
        },
        "Naive_Bayes": {
            "estimator": GaussianNB(),
            "params": {"var_smoothing": np.logspace(-11, -3, 15)},
            "search_type": "grid",
            "X": X_train, "y": y_train
        },
        "SVM": {
            "estimator": SVC(probability=False, class_weight="balanced", cache_size=1000, random_state=RANDOM_STATE),
            "params": [
                {"kernel": ["linear"], "C": [0.1, 1.0]},
                {"kernel": ["rbf"], "C": [1.0, 5.0], "gamma": ["scale", 0.01]}
            ],
            "search_type": "grid",
            "X": X_train_svm, "y": y_train_svm
        },
        "MLP_NeuralNet": {
            "estimator": MLPClassifier(
                activation="relu",
                solver="adam",
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=10,
                max_iter=150,
                random_state=RANDOM_STATE
            ),
            "params": {
                "hidden_layer_sizes": [(64,), (128, 64), (64, 32)],
                "alpha": [0.0001, 0.001, 0.01, 0.1],
                "learning_rate_init": [0.001, 0.005]
            },
            "search_type": "random",
            "n_iter": 8,
            "X": X_train, "y": y_train
        }
    }

    trained_models = {}
    cv_summary = {}

    for name, cfg in models_config.items():
        print(f"\n2. Ajustando hiperparámetros para: {name}...")
        start_time = time.time()

        if cfg["search_type"] == "grid":
            search = GridSearchCV(
                estimator=cfg["estimator"],
                param_grid=cfg["params"],
                scoring=scoring_metric,
                cv=cv,
                n_jobs=1,
                refit=True
            )
        else:
            search = RandomizedSearchCV(
                estimator=cfg["estimator"],
                param_distributions=cfg["params"],
                n_iter=cfg["n_iter"],
                scoring=scoring_metric,
                cv=cv,
                n_jobs=1,
                random_state=RANDOM_STATE,
                refit=True
            )

        search.fit(cfg["X"], cfg["y"])
        best_idx = search.best_index_
        mean_f1 = search.cv_results_["mean_test_score"][best_idx]
        std_f1 = search.cv_results_["std_test_score"][best_idx]

        if name == "SVM":
            best_svm = SVC(**search.best_params_, probability=True, class_weight="balanced", cache_size=1000, random_state=RANDOM_STATE)
            best_svm.fit(X_train_svm, y_train_svm)
            trained_models[name] = best_svm
        else:
            trained_models[name] = search.best_estimator_

        elapsed = time.time() - start_time
        cv_summary[name] = {
            "best_params": str(search.best_params_),
            "cv_mean_f1": round(float(mean_f1), 4),
            "cv_std_f1": round(float(std_f1), 4),
            "training_time_sec": round(elapsed, 2)
        }

        print(f"   Mejores hiperparámetros: {search.best_params_}")
        print(f"   F1 Validación Cruzada (5-fold): {mean_f1:.4f} ± {std_f1:.4f} ({elapsed:.1f}s)")

    joblib.dump(trained_models, "results/metrics/trained_models.joblib")
    with open("results/metrics/cv_results.json", "w") as f:
        json.dump(cv_summary, f, indent=4)

    print("\nEntrenamiento y validación cruzada completados.")

if __name__ == "__main__":
    main()