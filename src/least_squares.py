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


def least_squares(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(features.T @ features) @ features.T @ target


def plot_prediction(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    min_val = float(min(y_true.min(), y_pred.min()))
    max_val = float(max(y_true.max(), y_pred.max()))
    plt.plot([min_val, max_val], [min_val, max_val], "r--")
    plt.xlabel("True Price")
    plt.ylabel("Predicted Price")
    plt.title("Least Squares Prediction")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_least_squares(data_path: Path | None = None, results_dir: Path | None = None) -> dict[str, float]:
    base_dir = Path(__file__).resolve().parents[1]
    data_path = data_path or (base_dir / "data" / "house_data.csv")
    results_dir = results_dir or (base_dir / "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    features, target = load_data(data_path)
    features = add_bias(standardize(features))
    beta = least_squares(features, target)
    predictions = features @ beta

    mse = float(np.mean((target - predictions) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(target - predictions)))

    plot_prediction(target, predictions, results_dir / "least_squares_prediction.png")
    with open(results_dir / "least_squares_metrics.txt", "w", encoding="utf-8") as file:
        file.write(f"MSE: {mse:.6f}\nRMSE: {rmse:.6f}\nMAE: {mae:.6f}\n")

    print(f"Least squares beta: {beta}")
    print(f"Least squares MSE: {mse:.6f}")
    return {"mse": mse, "rmse": rmse, "mae": mae}


def main() -> None:
    run_least_squares()


if __name__ == "__main__":
    main()