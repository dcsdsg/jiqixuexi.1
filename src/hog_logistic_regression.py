# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


IMAGE_SIZE = (28, 28)
CELL_SIZE = (4, 4)
BLOCK_SIZE = (8, 8)
BLOCK_STRIDE = (4, 4)
NBINS = 9

LEARNING_RATE = 0.1
EPOCHS = 200
REG_LAMBDA = 1e-4
EPSILON = 1e-12


def build_hog_descriptor():
    import cv2

    return cv2.HOGDescriptor(IMAGE_SIZE, BLOCK_SIZE, BLOCK_STRIDE, CELL_SIZE, NBINS)


def iter_image_files(folder: Path) -> list[Path]:
    supported_suffixes = {".png", ".jpg", ".jpeg", ".bmp"}
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in supported_suffixes)


def extract_hog_feature(image_path: Path, hog_descriptor) -> np.ndarray:
    image = Image.open(image_path).convert("L")
    if image.size != IMAGE_SIZE:
        image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

    import cv2

    image_array = np.array(image, dtype=np.uint8)
    feature = hog_descriptor.compute(image_array)
    return feature.astype(np.float64).reshape(-1)


def load_dataset(data_dir: Path, class_a: str = "0", class_b: str = "1") -> tuple[np.ndarray, np.ndarray]:
    hog_descriptor = build_hog_descriptor()
    features: list[np.ndarray] = []
    labels: list[int] = []

    for folder_name, label in ((class_a, 0), (class_b, 1)):
        folder = data_dir / folder_name
        image_files = iter_image_files(folder)
        if not image_files:
            raise FileNotFoundError(f"No image files found in {folder}")

        for image_path in image_files:
            features.append(extract_hog_feature(image_path, hog_descriptor))
            labels.append(label)

    return np.vstack(features), np.array(labels, dtype=np.float64)


def compute_standardization_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def standardize_with_stats(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (features - mean) / std


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.clip(logits, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-logits))


def compute_sample_weights(labels: np.ndarray) -> np.ndarray:
    positive_count = max(float(np.sum(labels == 1.0)), 1.0)
    negative_count = max(float(np.sum(labels == 0.0)), 1.0)
    total_count = float(labels.size)
    positive_weight = total_count / (2.0 * positive_count)
    negative_weight = total_count / (2.0 * negative_count)
    return np.where(labels == 1.0, positive_weight, negative_weight)


def compute_loss_and_gradients(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    bias: float,
    sample_weights: np.ndarray,
    reg_lambda: float,
) -> tuple[float, np.ndarray, float]:
    logits = features @ weights + bias
    probabilities = np.clip(sigmoid(logits), EPSILON, 1.0 - EPSILON)
    errors = probabilities - labels
    weight_sum = float(sample_weights.sum())

    data_loss = -np.sum(
        sample_weights * (labels * np.log(probabilities) + (1.0 - labels) * np.log(1.0 - probabilities))
    ) / weight_sum
    reg_loss = 0.5 * reg_lambda * float(np.dot(weights, weights))

    grad_w = (features.T @ (sample_weights * errors)) / weight_sum + reg_lambda * weights
    grad_b = float(np.sum(sample_weights * errors) / weight_sum)
    return data_loss + reg_loss, grad_w, grad_b


def train_logistic_regression(
    features: np.ndarray,
    labels: np.ndarray,
    learning_rate: float,
    epochs: int,
    reg_lambda: float,
) -> tuple[np.ndarray, float, list[float]]:
    weights = np.zeros(features.shape[1], dtype=np.float64)
    bias = 0.0
    losses: list[float] = []
    sample_weights = compute_sample_weights(labels)

    for epoch in range(epochs):
        loss, grad_w, grad_b = compute_loss_and_gradients(features, labels, weights, bias, sample_weights, reg_lambda)
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b
        losses.append(loss)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:03d}/{epochs}, loss={loss:.6f}")

    return weights, bias, losses


def predict(features: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    probabilities = sigmoid(features @ weights + bias)
    return (probabilities >= 0.5).astype(np.int32)


def plot_loss_curve(losses: list[float], output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, color="tab:blue", linewidth=2)
    plt.title("HOG + Logistic Regression Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_hog_logistic_regression(data_root: Path | None = None, results_dir: Path | None = None, class_a: str = "0", class_b: str = "1") -> dict[str, float]:
    base_dir = Path(__file__).resolve().parents[1]
    data_root = data_root or (base_dir / "data" / "mnist_split" / "mnist_split")
    results_dir = results_dir or (base_dir / "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    train_dir = data_root / "train"
    test_dir = data_root / "test"

    train_features, train_labels = load_dataset(train_dir, class_a=class_a, class_b=class_b)
    test_features, test_labels = load_dataset(test_dir, class_a=class_a, class_b=class_b)

    train_mean, train_std = compute_standardization_stats(train_features)
    train_features = standardize_with_stats(train_features, train_mean, train_std)
    test_features = standardize_with_stats(test_features, train_mean, train_std)

    weights, bias, losses = train_logistic_regression(
        train_features,
        train_labels,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        reg_lambda=REG_LAMBDA,
    )

    train_predictions = predict(train_features, weights, bias)
    test_predictions = predict(test_features, weights, bias)
    train_accuracy = float(np.mean(train_predictions == train_labels))
    test_accuracy = float(np.mean(test_predictions == test_labels))

    plot_loss_curve(losses, results_dir / "hog_logistic_loss_curve.png")

    with open(results_dir / "hog_logistic_metrics.txt", "w", encoding="utf-8") as file:
        file.write(f"train_accuracy: {train_accuracy:.6f}\n")
        file.write(f"test_accuracy: {test_accuracy:.6f}\n")
        file.write(f"final_loss: {losses[-1]:.6f}\n")

    print(f"Loaded train samples: {train_labels.size}")
    print(f"Loaded test samples: {test_labels.size}")
    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    return {"train_accuracy": train_accuracy, "test_accuracy": test_accuracy, "final_loss": float(losses[-1])}


def main() -> None:
    run_hog_logistic_regression()


if __name__ == "__main__":
    main()