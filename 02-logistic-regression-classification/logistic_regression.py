"""
Logistic Regression from scratch — gradient descent on binary cross-entropy,
implemented with nothing but numpy, checked against scikit-learn, and compared
against the same model trained with the "wrong" loss (MSE) to show why the loss
function choice isn't arbitrary.

Run:
    python logistic_regression.py

See README.md for the math behind every formula referenced in the comments below.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write plots to file; don't require a display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(seed=42)
OUTPUT_DIR = Path(__file__).parent / "outputs"


class LogisticRegressionGD:
    """
    Model:      z = X @ w + b                                             (1)
                y_hat = sigmoid(z) = 1 / (1 + exp(-z))                     (2)
    Loss (BCE): J(w, b) = -(1/n) * Σ [y*log(y_hat) + (1-y)*log(1-y_hat)]   (3)
    Gradients:  dJ/dw = (1/n) * X.T @ (y_hat - y)                          (4)
                dJ/db = (1/n) * Σ (y_hat - y)                              (5)
    Update:     w <- w - lr * dJ/dw ,  b <- b - lr * dJ/db                 (6)

    See README.md "The math" section for the derivation of (4)/(5) from (3) — the
    sigmoid's own derivative cancels out of the chain rule, which is *why* the
    gradient has this simple form.

    loss="bce" (default) trains with cross-entropy, the statistically correct loss
    for this model (derived in the README from maximum likelihood). loss="mse"
    trains with mean-squared-error on the sigmoid output instead, so Part 3 of this
    script can empirically show why that's a worse choice, without changing anything
    else about the algorithm.
    """

    def __init__(
        self, learning_rate: float = 0.1, n_iterations: int = 1000, loss: str = "bce"
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.loss_type = loss
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)  # prevent exp() overflow on very confident predictions
        return 1.0 / (1.0 + np.exp(-z))

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        init_weights: np.ndarray | None = None,
        init_bias: float = 0.0,
    ) -> "LogisticRegressionGD":
        n_samples, n_features = X.shape
        # Zero-init by default. Part 3 below deliberately overrides this with a bad
        # starting point to demonstrate the vanishing-gradient failure mode of MSE.
        self.weights = np.zeros(n_features) if init_weights is None else init_weights.copy()
        self.bias = init_bias
        self.loss_history = []

        for _ in range(self.n_iterations):
            z = X @ self.weights + self.bias  # (1)
            y_hat = self._sigmoid(z)  # (2)

            if self.loss_type == "bce":
                eps = 1e-12
                y_hat_clipped = np.clip(y_hat, eps, 1 - eps)
                loss = -np.mean(
                    y * np.log(y_hat_clipped) + (1 - y) * np.log(1 - y_hat_clipped)
                )  # (3)
                dz = y_hat - y  # dJ/dz — see README derivation
            else:  # "mse": squared error on the sigmoid output, for comparison only
                error = y_hat - y
                loss = np.mean(error**2)
                dz = error * y_hat * (1 - y_hat)  # dJ/dz includes the sigmoid derivative

            self.loss_history.append(loss)

            dw = (1 / n_samples) * (X.T @ dz)  # (4), or its MSE analogue
            db = (1 / n_samples) * np.sum(dz)  # (5)

            self.weights -= self.learning_rate * dw  # (6)
            self.bias -= self.learning_rate * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


def make_synthetic_2d(n_per_class: int = 150):
    class0 = RNG.normal(loc=[-2, -2], scale=1.2, size=(n_per_class, 2))
    class1 = RNG.normal(loc=[2, 2], scale=1.2, size=(n_per_class, 2))
    X = np.vstack([class0, class1])
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)])
    return X, y


def run_synthetic_demo() -> None:
    print("=" * 70)
    print("PART 1 — Synthetic 2D data: decision boundary & loss curve")
    print("=" * 70)

    X, y = make_synthetic_2d()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegressionGD(learning_rate=0.1, n_iterations=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Learned: w={model.weights.round(3)}, b={model.bias:.3f}")
    print(f"Test accuracy: {acc:.3f}")

    plt.figure(figsize=(6, 4))
    plt.plot(model.loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Binary cross-entropy loss")
    plt.title("Loss curve — gradient descent converging")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_curve.png", dpi=120)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", alpha=0.7, edgecolor="k")
    x1_range = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)
    # Decision boundary: the line where p(y=1) = 0.5, i.e. z = w0*x0 + w1*x1 + b = 0.
    # Solve for x1 in terms of x0 to draw it: x1 = -(w0*x0 + b) / w1
    x2_boundary = -(model.weights[0] * x1_range + model.bias) / model.weights[1]
    plt.plot(x1_range, x2_boundary, color="black", linewidth=2, label="decision boundary (p=0.5)")
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.legend()
    plt.title("Decision boundary — linear in x, because z is linear in x")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "decision_boundary.png", dpi=120)
    plt.close()

    print(f"Saved plots to {OUTPUT_DIR}/loss_curve.png and decision_boundary.png")


def run_breast_cancer_demo() -> None:
    print()
    print("=" * 70)
    print("PART 2 — Breast cancer diagnosis: scratch vs. scikit-learn")
    print("=" * 70)

    data = load_breast_cancer()
    X, y = data.data, data.target  # 0 = malignant, 1 = benign

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize: same reason as project 01 — gradient descent needs comparable
    # feature scales to converge reliably (this dataset's features range from ~0.05
    # to ~2500 in raw units).
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    scratch_model = LogisticRegressionGD(learning_rate=0.1, n_iterations=2000)
    scratch_model.fit(X_train_scaled, y_train)
    y_pred_scratch = scratch_model.predict(X_test_scaled)

    sklearn_model = SklearnLogisticRegression(max_iter=2000)
    sklearn_model.fit(X_train_scaled, y_train)
    y_pred_sklearn = sklearn_model.predict(X_test_scaled)

    print(f"\n{'Method':<32}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}")
    print("-" * 70)
    for name, y_pred in [
        ("Scratch (gradient descent)", y_pred_scratch),
        ("scikit-learn LogisticRegression", y_pred_sklearn),
    ]:
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"{name:<32}{acc:>10.3f}{prec:>11.3f}{rec:>9.3f}{f1:>8.3f}")

    print(
        "\nScratch and sklearn should be close (not necessarily bit-for-bit — sklearn's "
        "solver is a different optimizer, L-BFGS by default, with its own regularization "
        "default, not plain gradient descent). Project 03 covers what precision/recall/F1 "
        "actually mean and when to care about each."
    )


def run_loss_comparison_demo() -> None:
    print()
    print("=" * 70)
    print("PART 3 — Why cross-entropy, not MSE: starting from a confidently wrong guess")
    print("=" * 70)

    X, y = make_synthetic_2d()
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Deliberately start both models from a BAD initialization: weights pointing
    # the wrong way, so the model begins by confidently predicting the wrong class
    # for most points (unlike the usual zero-init, which starts undecided at p=0.5
    # for everyone). This is the scenario where the two losses' gradients diverge
    # in behavior — see the printed explanation below.
    bad_init_weights = np.array([-8.0, -8.0])
    bad_init_bias = -2.0

    bce_model = LogisticRegressionGD(learning_rate=0.1, n_iterations=300, loss="bce")
    bce_model.fit(X_train, y_train, init_weights=bad_init_weights, init_bias=bad_init_bias)

    mse_model = LogisticRegressionGD(learning_rate=0.1, n_iterations=300, loss="mse")
    mse_model.fit(X_train, y_train, init_weights=bad_init_weights, init_bias=bad_init_bias)

    print(f"Same bad starting point, same learning rate, same data, 300 iterations:")
    print(f"  Cross-entropy loss: {bce_model.loss_history[0]:.3f} -> {bce_model.loss_history[-1]:.3f}")
    print(f"  MSE loss:           {mse_model.loss_history[0]:.3f} -> {mse_model.loss_history[-1]:.3f}")

    plt.figure(figsize=(7, 4))
    # Normalize each curve to its own starting value so both are visually comparable
    # on one axis (the two losses live on different scales, but "fraction of initial
    # loss remaining" is directly comparable).
    plt.plot(
        np.array(bce_model.loss_history) / bce_model.loss_history[0],
        label="Cross-entropy loss (normalized)",
    )
    plt.plot(
        np.array(mse_model.loss_history) / mse_model.loss_history[0],
        label="MSE loss (normalized)",
    )
    plt.xlabel("Iteration")
    plt.ylabel("Loss / initial loss")
    plt.title("Escaping a confidently-wrong start: cross-entropy vs. MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bce_vs_mse_convergence.png", dpi=120)
    plt.close()

    print(f"Saved plot to {OUTPUT_DIR}/bce_vs_mse_convergence.png")
    print(
        "\nCross-entropy escapes the bad start and converges. MSE barely moves at all — "
        "it's stuck. Why: MSE's gradient carries an extra y_hat*(1-y_hat) factor (the "
        "sigmoid's own derivative). When the model is confidently wrong, y_hat is near "
        "0 or 1, so that factor is near 0 too — the gradient vanishes at exactly the "
        "moment the model most needs a big correction. Cross-entropy's gradient "
        "(y_hat - y) has no such term: it stays proportional to how wrong the "
        "prediction is, confident or not, which is what lets it recover."
    )


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    run_synthetic_demo()
    run_breast_cancer_demo()
    run_loss_comparison_demo()
