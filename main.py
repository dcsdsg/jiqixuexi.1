# -*- coding: utf-8 -*-
import argparse
from datetime import datetime
from pathlib import Path

from data_preprocess import preprocess_titanic
from src.gradient_descent import run_gradient_descent
from src.hog_logistic_regression import run_hog_logistic_regression
from src.hog_pedestrian_demo import real_time_pedestrian_masking
from src.least_squares import run_least_squares
from src.logistic_regression_titanic import run_titanic_logistic_regression
from src.svm import test_svm, train_svm


def main() -> None:
    parser = argparse.ArgumentParser(description="Machine learning project entry point")
    parser.add_argument("--algo", required=True, type=str, help="svm / logreg / ls / gd / hog_lr / hog_demo")
    parser.add_argument("--data", required=True, type=str, help="titanic / house / mnist_split / demo")
    parser.add_argument("--process", required=True, type=str, help="train / test / run")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    models_dir = base_dir / "models"
    results_dir = base_dir / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.algo == "svm" and args.data == "titanic":
        model_path = models_dir / f"svm_titanic_{datetime.now().strftime('%Y%m%d')}.pkl"

        if args.process == "train":
            print("Starting Titanic preprocessing...")
            x_train, x_test, y_train, y_test = preprocess_titanic()
            print("Training SVM model...")
            train_svm(x_train, y_train, model_save_path=str(model_path))
            print(f"Model saved to {model_path}")
        elif args.process == "test":
            print("Loading Titanic data and evaluating SVM model...")
            x_train, x_test, y_train, y_test = preprocess_titanic()
            test_svm(x_test, y_test, model_path=str(model_path))
        else:
            print("Unsupported process for svm. Use train or test.")
        return

    if args.algo == "logreg" and args.data == "titanic":
        run_titanic_logistic_regression(results_dir=results_dir)
        return

    if args.algo == "ls" and args.data == "house":
        run_least_squares(results_dir=results_dir)
        return

    if args.algo == "gd" and args.data == "house":
        run_gradient_descent(results_dir=results_dir)
        return

    if args.algo == "hog_lr" and args.data == "mnist_split":
        run_hog_logistic_regression(results_dir=results_dir)
        return

    if args.algo == "hog_demo" and args.data == "demo":
        real_time_pedestrian_masking()
        return

    print("Unsupported algorithm/data combination.")


if __name__ == "__main__":
    main()
