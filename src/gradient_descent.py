# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    features = df[["x1", "x2", "x3", "x4"]].to_numpy(dtype=np.float64)
    target = df["y"].to_numpy(dtype=np.float64)
    return features, target


def standardize(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-12] = 1.0
    return (features - mean) / std


def add_bias(features: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((features.shape[0], 1)), features])


def compute_loss(features: np.ndarray, target: np.ndarray, beta: np.ndarray) -> float:
    prediction = features @ beta
    return float(np.mean((prediction - target) ** 2))


def batch_gradient_descent(features: np.ndarray, target: np.ndarray, lr: float = 1e-2, epochs: int = 1000) -> tuple[np.ndarray, list[float]]:
    beta = np.zeros(features.shape[1], dtype=np.float64)
    loss_history: list[float] = []
    sample_count = features.shape[0]

    for _ in range(epochs):
        prediction = features @ beta
        gradient = (2.0 / sample_count) * (features.T @ (prediction - target))
        beta -= lr * gradient
        loss_history.append(compute_loss(features, target, beta))

    return beta, loss_history


def stochastic_gradient_descent(features: np.ndarray, target: np.ndarray, lr: float = 5e-3, epochs: int = 10) -> tuple[np.ndarray, list[float]]:
    beta = np.zeros(features.shape[1], dtype=np.float64)
    loss_history: list[float] = []

    for _ in range(epochs):
        for index in range(features.shape[0]):
            xi = features[index]
            yi = target[index]
            prediction = float(xi @ beta)
            gradient = 2.0 * xi * (prediction - yi)
            beta -= lr * gradient
        loss_history.append(compute_loss(features, target, beta))

    return beta, loss_history


def mini_batch_gradient_descent(features: np.ndarray, target: np.ndarray, lr: float = 1e-2, epochs: int = 100, batch_size: int = 16) -> tuple[np.ndarray, list[float]]:
    beta = np.zeros(features.shape[1], dtype=np.float64)
    loss_history: list[float] = []
    sample_count = features.shape[0]

    for _ in range(epochs):
        indices = np.random.permutation(sample_count)
        shuffled_features = features[indices]
        shuffled_target = target[indices]

        for start in range(0, sample_count, batch_size):
            batch_features = shuffled_features[start:start + batch_size]
            batch_target = shuffled_target[start:start + batch_size]
            prediction = batch_features @ beta
            gradient = (2.0 / len(batch_features)) * (batch_features.T @ (prediction - batch_target))
            beta -= lr * gradient

        loss_history.append(compute_loss(features, target, beta))

    return beta, loss_history


def plot_loss(loss_dict: dict[str, list[float]], output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for name, losses in loss_dict.items():
        plt.plot(losses, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Gradient Descent Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_prediction(y_true: np.ndarray, y_pred: np.ndarray, title: str, output_path: Path) -> None:
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    min_val = float(min(y_true.min(), y_pred.min()))
    max_val = float(max(y_true.max(), y_pred.max()))
    plt.plot([min_val, max_val], [min_val, max_val], "r--")
    plt.xlabel("True Price")
    plt.ylabel("Predicted Price")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_gradient_descent(data_path: Path | None = None, results_dir: Path | None = None) -> dict[str, float]:
    base_dir = Path(__file__).resolve().parents[1]
    data_path = data_path or (base_dir / "data" / "house_data.csv")
    results_dir = results_dir or (base_dir / "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    features, target = load_data(data_path)
    features = add_bias(standardize(features))

    beta_bgd, loss_bgd = batch_gradient_descent(features, target)
    beta_sgd, loss_sgd = stochastic_gradient_descent(features, target)
    beta_mgd, loss_mgd = mini_batch_gradient_descent(features, target)

    plot_loss(
        {"BGD": loss_bgd, "SGD": loss_sgd, "Mini-batch": loss_mgd},
        results_dir / "gradient_descent_loss_curve.png",
    )

    pred_bgd = features @ beta_bgd
    pred_sgd = features @ beta_sgd
    pred_mgd = features @ beta_mgd

    plot_prediction(target, pred_bgd, "BGD Prediction", results_dir / "gradient_bgd_prediction.png")
    plot_prediction(target, pred_sgd, "SGD Prediction", results_dir / "gradient_sgd_prediction.png")
    plot_prediction(target, pred_mgd, "Mini-batch Prediction", results_dir / "gradient_mini_batch_prediction.png")

    metrics = {
        "bgd_mse": float(np.mean((target - pred_bgd) ** 2)),
        "sgd_mse": float(np.mean((target - pred_sgd) ** 2)),
        "mgd_mse": float(np.mean((target - pred_mgd) ** 2)),
    }
    with open(results_dir / "gradient_descent_metrics.txt", "w", encoding="utf-8") as file:
        for key, value in metrics.items():
            file.write(f"{key}: {value:.6f}\n")

    print("Gradient descent experiments finished.")
    return metrics


def main() -> None:
    run_gradient_descent()


if __name__ == "__main__":
    main()