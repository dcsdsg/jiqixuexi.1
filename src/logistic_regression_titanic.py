# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from data_preprocess import preprocess_titanic


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


def train_mini_batch(
    features: np.ndarray,
    target: np.ndarray,
    learning_rate: float = 0.1,
    batch_size: int = 32,
    epochs: int = 200,
    reg_lambda: float = 0.1,
) -> tuple[np.ndarray, float, list[float]]:
    sample_count, feature_count = features.shape
    weights = np.zeros(feature_count, dtype=np.float64)
    bias = 0.0
    losses: list[float] = []

    for _ in range(epochs):
        shuffled_indices = np.random.permutation(sample_count)
        shuffled_features = features[shuffled_indices]
        shuffled_target = target[shuffled_indices]

        for start in range(0, sample_count, batch_size):
            batch_features = shuffled_features[start:start + batch_size]
            batch_target = shuffled_target[start:start + batch_size]

            logits = batch_features @ weights + bias
            probabilities = sigmoid(logits)
            batch_size_actual = batch_features.shape[0]

            gradient_w = (batch_features.T @ (probabilities - batch_target)) / batch_size_actual + (reg_lambda / sample_count) * weights
            gradient_b = float(np.sum(probabilities - batch_target) / batch_size_actual)

            weights -= learning_rate * gradient_w
            bias -= learning_rate * gradient_b

        logits = features @ weights + bias
        probabilities = np.clip(sigmoid(logits), 1e-15, 1 - 1e-15)
        loss = -np.mean(target * np.log(probabilities) + (1 - target) * np.log(1 - probabilities))
        loss += (reg_lambda / (2 * sample_count)) * float(np.sum(weights ** 2))
        losses.append(float(loss))

    return weights, bias, losses


def manual_predict(features: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    probabilities = sigmoid(features @ weights + bias)
    return (probabilities >= 0.5).astype(int)


def plot_losses(losses: list[float], output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.title("Manual Logistic Regression Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_titanic_logistic_regression(results_dir: Path | None = None) -> dict[str, float]:
    base_dir = Path(__file__).resolve().parents[1]
    results_dir = results_dir or (base_dir / "results")
    models_dir = base_dir / "models"
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    x_train, x_test, y_train, y_test = preprocess_titanic()

    sklearn_model = LogisticRegression(max_iter=1000, random_state=42)
    sklearn_model.fit(x_train, y_train)
    joblib.dump(sklearn_model, models_dir / "logistic_regression_titanic.pkl")

    train_pred_sklearn = sklearn_model.predict(x_train)
    test_pred_sklearn = sklearn_model.predict(x_test)

    weights, bias, losses = train_mini_batch(x_train, y_train)
    plot_losses(losses, results_dir / "logistic_regression_loss_curve.png")

    train_pred_manual = manual_predict(x_train, weights, bias)
    test_pred_manual = manual_predict(x_test, weights, bias)

    metrics = {
        "sklearn_train_accuracy": float(accuracy_score(y_train, train_pred_sklearn)),
        "sklearn_test_accuracy": float(accuracy_score(y_test, test_pred_sklearn)),
        "manual_train_accuracy": float(accuracy_score(y_train, train_pred_manual)),
        "manual_test_accuracy": float(accuracy_score(y_test, test_pred_manual)),
        "manual_final_loss": float(losses[-1]),
    }

    with open(results_dir / "logistic_regression_metrics.txt", "w", encoding="utf-8") as file:
        for key, value in metrics.items():
            file.write(f"{key}: {value:.6f}\n")

    print("Titanic logistic regression finished.")
    return metrics


def main() -> None:
    run_titanic_logistic_regression()


if __name__ == "__main__":
    main()