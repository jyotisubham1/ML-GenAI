"""
Linear Regression from scratch — batch gradient descent implemented with nothing but
numpy, checked against the closed-form normal equation and against scikit-learn.

Run:
    python linear_regression.py

See README.md for the math behind every formula referenced in the comments below.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write plots to file; don't require a display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(seed=42)
OUTPUT_DIR = Path(__file__).parent / "outputs"


class LinearRegressionGD:
    """
    Fits y_hat = X @ w + b by minimizing Mean Squared Error with batch gradient descent.

    Model:      y_hat = X @ w + b                                  (1)
    Loss (MSE): J(w, b) = (1/n) * sum((y_hat_i - y_i)^2)            (2)
    Gradients:  dJ/dw = (2/n) * X.T @ (y_hat - y)                   (3)
                dJ/db = (2/n) * sum(y_hat - y)                      (4)
    Update:     w <- w - lr * dJ/dw ,  b <- b - lr * dJ/db          (5)

    See README.md "The math" section for the derivation of (3) and (4) from (2).
    """

    def __init__(self, learning_rate: float = 0.1, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iterations):
            y_pred = X @ self.weights + self.bias  # (1)

            error = y_pred - y
            self.loss_history.append(np.mean(error**2))  # (2)

            dw = (2 / n_samples) * (X.T @ error)  # (3)
            db = (2 / n_samples) * np.sum(error)  # (4)

            self.weights -= self.learning_rate * dw  # (5)
            self.bias -= self.learning_rate * db  # (5)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias


def normal_equation(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Closed-form solution that minimizes MSE directly: w = (X^T X)^-1 X^T y.
    Derived by setting dJ/dw = 0 and solving (see README). Only practical when
    X^T X is invertible and small enough to invert (O(d^3) in feature count d);
    gradient descent scales to far more features and to models with no closed form
    (like neural nets), which is why we learn both.
    """
    X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])
    theta = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
    bias, weights = theta[0], theta[1:]
    return weights, bias


def run_synthetic_demo() -> None:
    print("=" * 70)
    print("PART 1 — Synthetic 1D data: fit a line, watch the loss curve")
    print("=" * 70)

    n_samples = 200
    X = RNG.uniform(0, 10, size=(n_samples, 1))
    true_weight, true_bias = 3.5, -2.0
    noise = RNG.normal(0, 2.0, size=n_samples)
    y = true_weight * X[:, 0] + true_bias + noise

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegressionGD(learning_rate=0.01, n_iterations=1000)
    model.fit(X_train, y_train)

    print(f"True function:  y = {true_weight} * x + {true_bias}")
    print(f"Learned (GD):   y = {model.weights[0]:.3f} * x + {model.bias:.3f}")

    y_pred = model.predict(X_test)
    print(
        f"Test MSE: {mean_squared_error(y_test, y_pred):.3f}   "
        f"R^2: {r2_score(y_test, y_pred):.3f}"
    )

    plt.figure(figsize=(6, 4))
    plt.plot(model.loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("MSE loss")
    plt.title("Loss curve — gradient descent converging")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_curve.png", dpi=120)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.scatter(X_train, y_train, alpha=0.5, label="train data")
    x_line = np.linspace(0, 10, 100).reshape(-1, 1)
    plt.plot(x_line, model.predict(x_line), color="red", label="learned fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Fitted line vs. data")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fitted_line.png", dpi=120)
    plt.close()

    print(f"Saved plots to {OUTPUT_DIR}/loss_curve.png and fitted_line.png")


def run_california_housing_demo() -> None:
    print()
    print("=" * 70)
    print("PART 2 — California housing: scratch GD vs. normal equation vs. sklearn")
    print("=" * 70)

    housing = fetch_california_housing()
    X, y = housing.data, housing.target  # target: median house value, $100,000s

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features: gradient descent converges far more reliably (and faster)
    # when features share a scale. Without this, a feature like "population" (scale:
    # thousands) dominates the gradient over "rooms per household" (scale: single
    # digits), even if both matter equally to the prediction. The normal equation
    # doesn't strictly need this, but we scale for both so the comparison is apples
    # to apples.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    scratch_model = LinearRegressionGD(learning_rate=0.1, n_iterations=1000)
    scratch_model.fit(X_train_scaled, y_train)
    results["Scratch (gradient descent)"] = scratch_model.predict(X_test_scaled)

    weights_ne, bias_ne = normal_equation(X_train_scaled, y_train)
    results["Normal equation (closed form)"] = X_test_scaled @ weights_ne + bias_ne

    sklearn_model = SklearnLinearRegression()
    sklearn_model.fit(X_train_scaled, y_train)
    results["scikit-learn LinearRegression"] = sklearn_model.predict(X_test_scaled)

    print(f"\n{'Method':<32}{'MSE':>10}{'R^2':>10}")
    print("-" * 52)
    for name, y_pred in results.items():
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{name:<32}{mse:>10.4f}{r2:>10.4f}")

    print(
        "\nAll three should land within a hair of each other: gradient descent is an "
        "iterative approximation converging toward the same minimum the normal "
        "equation reaches in one algebraic step, and scikit-learn's LinearRegression "
        "solves the same least-squares problem with a more numerically stable routine "
        "(SVD-based) under the hood."
    )


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    run_synthetic_demo()
    run_california_housing_demo()
