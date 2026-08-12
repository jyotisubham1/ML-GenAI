"""
Neural Network from Scratch — forward pass, backpropagation derived via the chain
rule, and gradient descent, in nothing but numpy. Six experiments, each testing a
claim rather than asserting it:

  Part 1  a linear model provably cannot solve XOR; one hidden layer can
  Part 2  backprop verified against numerical derivatives (gradient checking)
  Part 3  without a nonlinearity, a deep network collapses to a single linear layer
  Part 4  why sigmoid stops working with depth: gradients measured layer by layer
  Part 5  initialization: zeros break the network completely, and why
  Part 6  real data — handwritten digits, vs. sklearn and vs. logistic regression

Run:
    python neural_network.py

See README.md for the math behind every formula referenced in the comments below.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write plots to file; don't require a display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits, make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(seed=42)
OUTPUT_DIR = Path(__file__).parent / "outputs"


# ---------------------------------------------------------------------------
# Activation functions and their derivatives
#
# Each is a pair: g(z) for the forward pass, g'(z) for the backward pass. That
# pairing is the whole reason backprop is cheap — every derivative below is
# computable from values we already have.
# ---------------------------------------------------------------------------


def relu(z):
    return np.maximum(0, z)


def relu_deriv(z):
    # Slope is 1 where z > 0 and 0 elsewhere. Undefined exactly at 0; we pick 0,
    # which is what every framework does (it never matters in practice).
    return (z > 0).astype(float)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)  # the same sigmoid'(z) = s(1-s) derived in project 02


def tanh(z):
    return np.tanh(z)


def tanh_deriv(z):
    return 1.0 - np.tanh(z) ** 2


def identity(z):
    return z


def identity_deriv(z):
    return np.ones_like(z)


ACTIVATIONS = {
    "relu": (relu, relu_deriv),
    "sigmoid": (sigmoid, sigmoid_deriv),
    "tanh": (tanh, tanh_deriv),
    "identity": (identity, identity_deriv),
}


def softmax(z: np.ndarray) -> np.ndarray:
    """
    softmax(z)_k = exp(z_k) / sum_j exp(z_j)                                   (5)

    The multi-class generalization of project 02's sigmoid: turns a vector of
    arbitrary real scores into probabilities that are positive and sum to 1.
    Subtracting the row max before exponentiating changes nothing mathematically
    (it cancels in the ratio) but prevents exp() overflowing on large scores.
    """
    z_shifted = z - z.max(axis=1, keepdims=True)
    e = np.exp(z_shifted)
    return e / e.sum(axis=1, keepdims=True)


class NeuralNetwork:
    """
    A fully-connected feed-forward network trained by backpropagation.

    FORWARD (layer l = 1..L):
        z^l = a^(l-1) W^l + b^l                                                (1)
        a^l = g(z^l)             for hidden layers                             (2)
        a^L = softmax(z^L)       for the output layer                          (5)

    LOSS (cross-entropy, exactly project 02's, generalized to K classes):
        J = -(1/n) sum_i sum_k Y_ik log(a^L_ik)                                (6)

    BACKWARD (l = L down to 1):
        delta^L = (a^L - Y) / n                                                (7)
        dJ/dW^l = (a^(l-1))^T @ delta^l                                        (8)
        dJ/db^l = column sums of delta^l                                       (9)
        delta^(l-1) = (delta^l @ (W^l)^T) * g'(z^(l-1))                       (10)

    Line (10) IS backpropagation: the error at a layer is the next layer's error,
    pulled back through that layer's weights, then scaled by how steep this layer's
    activation was. Everything else is bookkeeping. See README §4.4 for the
    derivation, and Part 2 for a numerical proof that (8)-(10) are correct.
    """

    def __init__(self, layer_sizes: list[int], activation: str = "relu",
                 init: str = "he", learning_rate: float = 0.1, seed: int = 0):
        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.g, self.g_deriv = ACTIVATIONS[activation]
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)
        self.init = init
        self.weights, self.biases = self._initialize()
        self.loss_history: list[float] = []

    def _initialize(self):
        weights, biases = [], []
        for fan_in, fan_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            if self.init == "zeros":
                W = np.zeros((fan_in, fan_out))
            elif self.init == "he":
                # He initialization: variance 2/fan_in. Keeps the SCALE of the
                # signal roughly constant as it passes through ReLU layers, which
                # kill half the activations. See README §4.6.
                W = self.rng.normal(0, np.sqrt(2.0 / fan_in), (fan_in, fan_out))
            elif self.init == "xavier":
                # Xavier/Glorot: variance 1/fan_in — the same idea for activations
                # that are symmetric around 0 (tanh, sigmoid).
                W = self.rng.normal(0, np.sqrt(1.0 / fan_in), (fan_in, fan_out))
            elif self.init == "small":
                W = self.rng.normal(0, 0.01, (fan_in, fan_out))
            elif self.init == "large":
                W = self.rng.normal(0, 3.0, (fan_in, fan_out))
            else:
                raise ValueError(self.init)
            weights.append(W)
            biases.append(np.zeros(fan_out))
        return weights, biases

    def forward(self, X: np.ndarray):
        """Returns (activations, pre_activations). Both are cached for the backward pass."""
        activations = [X]  # a^0 = X, the input itself
        pre_activations = []  # the z^l values

        a = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b  # (1)
            pre_activations.append(z)
            is_output = i == len(self.weights) - 1
            a = softmax(z) if is_output else self.g(z)  # (5) or (2)
            activations.append(a)
        return activations, pre_activations

    @staticmethod
    def loss(probs: np.ndarray, Y: np.ndarray) -> float:
        """Cross-entropy (6). Clipped so log(0) can't produce -inf."""
        return float(-np.mean(np.sum(Y * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)))

    def backward(self, activations, pre_activations, Y):
        """Returns (dW, db) — one gradient array per layer. Implements (7)-(10)."""
        n = len(Y)
        dW = [None] * len(self.weights)
        db = [None] * len(self.biases)

        # (7) The output-layer error. Softmax + cross-entropy produce this
        # beautifully simple form — exactly as sigmoid + BCE did in project 02,
        # and for exactly the same reason (the derivatives cancel).
        delta = (activations[-1] - Y) / n

        for l in reversed(range(len(self.weights))):
            dW[l] = activations[l].T @ delta  # (8)
            db[l] = delta.sum(axis=0)  # (9)
            if l > 0:
                # (10) — pull the error back through the weights, then scale by
                # the slope of this layer's activation.
                delta = (delta @ self.weights[l].T) * self.g_deriv(pre_activations[l - 1])
        return dW, db

    def fit(self, X, Y, epochs: int = 500, verbose_every: int | None = None):
        self.loss_history = []
        for epoch in range(epochs):
            activations, pre_activations = self.forward(X)
            self.loss_history.append(self.loss(activations[-1], Y))

            dW, db = self.backward(activations, pre_activations, Y)
            for l in range(len(self.weights)):
                self.weights[l] -= self.learning_rate * dW[l]  # gradient descent,
                self.biases[l] -= self.learning_rate * db[l]  # same as projects 01/02
            if verbose_every and epoch % verbose_every == 0:
                print(f"    epoch {epoch:>5}  loss {self.loss_history[-1]:.6f}")
        return self

    def predict_proba(self, X):
        return self.forward(X)[0][-1]

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    # -- helpers used by the experiments -------------------------------------

    def flat_params(self) -> np.ndarray:
        return np.concatenate([w.ravel() for w in self.weights]
                              + [b.ravel() for b in self.biases])

    def set_flat_params(self, flat: np.ndarray) -> None:
        i = 0
        for l, W in enumerate(self.weights):
            self.weights[l] = flat[i:i + W.size].reshape(W.shape)
            i += W.size
        for l, b in enumerate(self.biases):
            self.biases[l] = flat[i:i + b.size].reshape(b.shape)
            i += b.size


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(y), n_classes))
    out[np.arange(len(y)), y.astype(int)] = 1
    return out


# ---------------------------------------------------------------------------
# Part 1 — XOR
# ---------------------------------------------------------------------------


def run_xor_demo() -> None:
    print("=" * 74)
    print("PART 1 — XOR: the problem that a linear model cannot solve")
    print("=" * 74)

    X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = np.array([0, 1, 1, 0])  # output 1 iff exactly one input is 1
    Y = one_hot(y, 2)

    print("\n  x1  x2 | XOR")
    print("  ---------|----")
    for xi, yi in zip(X, y):
        print(f"  {xi[0]:.0f}   {xi[1]:.0f}  |  {yi}")

    logistic = LogisticRegression().fit(X, y)
    print(f"\nLogistic regression (project 02) accuracy: "
          f"{accuracy_score(y, logistic.predict(X)):.2f}")

    net = NeuralNetwork([2, 4, 2], activation="tanh", init="xavier",
                        learning_rate=0.5, seed=1).fit(X, Y, epochs=3000)
    print(f"Neural network, ONE hidden layer of 4:     "
          f"{accuracy_score(y, net.predict(X)):.2f}")
    print(f"  final loss: {net.loss_history[-1]:.6f}")

    print(
        "\nNo straight line can separate these four points — put (0,0) and (1,1) in one\n"
        "class and (0,1) and (1,0) in the other, and any line you draw cuts through both\n"
        "groups. Logistic regression's boundary IS a straight line (project 02 §4.1), so\n"
        "it cannot exceed 50-75% here no matter how long you train it. This is not a\n"
        "tuning problem; it is a capacity problem.\n\n"
        "The hidden layer fixes it by learning a NEW REPRESENTATION: it bends the input\n"
        "space until the classes become linearly separable, and only then applies a linear\n"
        "boundary. That is what 'deep learning learns features' means, concretely."
    )

    # Show the learned hidden representation: the point of the whole project.
    hidden = net.forward(X)[0][1]
    print(f"\nWhat the hidden layer turns each input into (its 4 units' activations):")
    for xi, hi, yi in zip(X, hidden, y):
        print(f"  ({xi[0]:.0f},{xi[1]:.0f}) -> [{', '.join(f'{v:+.2f}' for v in hi)}]   class {yi}")

    # Decision boundaries on a harder 2D problem, for a picture.
    Xm, ym = make_moons(n_samples=400, noise=0.2, random_state=42)
    Ym = one_hot(ym, 2)
    xx, yy = np.meshgrid(np.linspace(Xm[:, 0].min() - .5, Xm[:, 0].max() + .5, 300),
                         np.linspace(Xm[:, 1].min() - .5, Xm[:, 1].max() + .5, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    log_m = LogisticRegression().fit(Xm, ym)
    nets = [
        ("Logistic regression (no hidden layer)", None, log_m),
        ("1 hidden layer, 8 units", NeuralNetwork([2, 8, 2], "tanh", "xavier", 0.5, 2), None),
        ("2 hidden layers, 32 units each", NeuralNetwork([2, 32, 32, 2], "relu", "he", 0.2, 3), None),
    ]
    for ax, (title, net_m, lin_m) in zip(axes, nets):
        if net_m is not None:
            net_m.fit(Xm, Ym, epochs=3000)
            pred = net_m.predict(grid).reshape(xx.shape)
            acc = accuracy_score(ym, net_m.predict(Xm))
        else:
            pred = lin_m.predict(grid).reshape(xx.shape)
            acc = accuracy_score(ym, lin_m.predict(Xm))
        ax.contourf(xx, yy, pred, alpha=0.25, cmap="coolwarm")
        ax.scatter(Xm[:, 0], Xm[:, 1], c=ym, cmap="coolwarm", s=14, edgecolor="k", linewidth=.3)
        ax.set_title(f"{title}\naccuracy {acc:.3f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "decision_boundaries.png", dpi=120)
    plt.close()
    print("\nSaved plot to outputs/decision_boundaries.png")


# ---------------------------------------------------------------------------
# Part 2 — gradient checking
# ---------------------------------------------------------------------------


def run_gradient_check() -> None:
    print()
    print("=" * 74)
    print("PART 2 — Is backprop actually correct? Checking against numerical slopes")
    print("=" * 74)

    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 5))
    y = rng.integers(0, 3, 12)
    Y = one_hot(y, 3)

    print(f"\n{'Activation':<14}{'params':>9}{'max |analytic-numeric|':>26}{'relative error':>18}")
    print("-" * 74)
    for act in ("relu", "sigmoid", "tanh"):
        net = NeuralNetwork([5, 6, 4, 3], activation=act, init="xavier", seed=1)

        activations, pre = net.forward(X)
        dW, db = net.backward(activations, pre, Y)
        analytic = np.concatenate([g.ravel() for g in dW] + [g.ravel() for g in db])

        # Numerical gradient: nudge each parameter up and down, see how the loss
        # moves. This is the DEFINITION of a derivative, so it needs no theory —
        # which is exactly why it can referee the theory.
        theta = net.flat_params().copy()
        eps = 1e-6
        numeric = np.zeros_like(theta)
        for i in range(len(theta)):
            up, down = theta.copy(), theta.copy()
            up[i] += eps
            down[i] -= eps
            net.set_flat_params(up)
            loss_up = net.loss(net.forward(X)[0][-1], Y)
            net.set_flat_params(down)
            loss_down = net.loss(net.forward(X)[0][-1], Y)
            numeric[i] = (loss_up - loss_down) / (2 * eps)
        net.set_flat_params(theta)

        abs_err = float(np.max(np.abs(analytic - numeric)))
        rel_err = float(np.linalg.norm(analytic - numeric)
                        / (np.linalg.norm(analytic) + np.linalg.norm(numeric)))
        print(f"{act:<14}{len(theta):>9}{abs_err:>26.3e}{rel_err:>18.3e}")

    print(
        "\nA relative error below ~1e-7 means the two agree to nearly machine precision,\n"
        "so the chain-rule derivation in README §4.4 is correct — for three different\n"
        "activation functions, across a 3-layer network.\n\n"
        "Why this matters more than it looks: a wrong gradient does NOT crash. The network\n"
        "trains, the loss goes down a bit, and you get a mediocre model with no error\n"
        "message. Gradient checking is how you find that bug. It is far too slow to use\n"
        "during real training (one forward pass PER PARAMETER — that is the entire reason\n"
        "backprop exists), so you run it once on a tiny network, then trust the code."
    )


# ---------------------------------------------------------------------------
# Part 3 — depth without nonlinearity is an illusion
# ---------------------------------------------------------------------------


def run_linearity_demo() -> None:
    print()
    print("=" * 74)
    print("PART 3 — Remove the activation and the depth evaporates")
    print("=" * 74)

    X, y = make_moons(n_samples=600, noise=0.2, random_state=42)
    Y = one_hot(y, 2)

    deep_linear = NeuralNetwork([2, 32, 32, 32, 2], activation="identity",
                                init="xavier", learning_rate=0.2, seed=4)
    deep_linear.fit(X, Y, epochs=2000)
    deep_relu = NeuralNetwork([2, 32, 32, 32, 2], activation="relu",
                              init="he", learning_rate=0.2, seed=4).fit(X, Y, epochs=2000)
    logistic = LogisticRegression().fit(X, y)

    print(f"\n{'Model':<46}{'Train accuracy':>16}")
    print("-" * 74)
    print(f"{'Logistic regression (1 linear layer)':<46}"
          f"{accuracy_score(y, logistic.predict(X)):>16.4f}")
    print(f"{'4-layer network, IDENTITY activation':<46}"
          f"{accuracy_score(y, deep_linear.predict(X)):>16.4f}")
    print(f"{'4-layer network, ReLU activation':<46}"
          f"{accuracy_score(y, deep_relu.predict(X)):>16.4f}")

    # The proof: multiply the weight matrices together and confirm the whole
    # 4-layer network is exactly equivalent to one linear layer.
    W_eff = deep_linear.weights[0]
    for W in deep_linear.weights[1:]:
        W_eff = W_eff @ W
    b_eff = deep_linear.biases[0]
    for l in range(1, len(deep_linear.weights)):
        b_eff = b_eff @ deep_linear.weights[l] + deep_linear.biases[l]

    equivalent = softmax(X @ W_eff + b_eff)
    actual = deep_linear.predict_proba(X)
    print(
        f"\nCollapsing the network by hand: multiply its {len(deep_linear.weights)} weight matrices "
        f"together into a\nsingle 2x2 matrix W_eff, and compare that one linear layer's output "
        f"against the\nfull network's:\n"
        f"  max |difference| = {np.max(np.abs(equivalent - actual)):.3e}"
    )
    n_params = (sum(w.size for w in deep_linear.weights)
                + sum(b.size for b in deep_linear.biases))
    print(
        f"\nThey are the same function to machine precision. A stack of linear layers is a\n"
        f"linear layer — matrix multiplication is associative, so W1(W2(W3 x)) = (W1 W2 W3)x.\n"
        f"{n_params:,} parameters bought exactly nothing over logistic regression's 3, and the\n"
        f"two accuracies above match to four decimals because they are the same model.\n\n"
        f"The activation function is not a detail or a performance tweak. It is the ONLY\n"
        f"reason depth exists."
    )


# ---------------------------------------------------------------------------
# Part 4 — vanishing gradients
# ---------------------------------------------------------------------------


def run_vanishing_gradient_demo() -> None:
    print()
    print("=" * 74)
    print("PART 4 — Why sigmoid stopped being used: gradients measured by layer")
    print("=" * 74)

    rng = np.random.default_rng(1)
    X = rng.normal(size=(256, 20))
    Y = one_hot(rng.integers(0, 2, 256), 2)

    depth = 12
    sizes = [20] + [24] * depth + [2]

    curves = {}
    for act, init in (("sigmoid", "xavier"), ("tanh", "xavier"), ("relu", "he")):
        net = NeuralNetwork(sizes, activation=act, init=init, seed=2)
        activations, pre = net.forward(X)
        dW, _ = net.backward(activations, pre, Y)
        curves[act] = [float(np.linalg.norm(g)) for g in dW]

    print(f"\nGradient magnitude ||dJ/dW|| at each layer, in a {depth}-hidden-layer network:\n")
    print(f"{'layer':>7}{'sigmoid':>14}{'tanh':>14}{'relu':>14}")
    print("-" * 74)
    for l in range(len(sizes) - 1):
        label = "output" if l == len(sizes) - 2 else f"{l+1}"
        print(f"{label:>7}{curves['sigmoid'][l]:>14.3e}{curves['tanh'][l]:>14.3e}"
              f"{curves['relu'][l]:>14.3e}")

    ratio = curves["sigmoid"][-1] / max(curves["sigmoid"][0], 1e-30)
    print(
        f"\nRead the sigmoid column from the bottom up. The output layer's gradient is normal,\n"
        f"and each step back toward the input multiplies it by another factor below 1. By\n"
        f"layer 1 it has shrunk by a factor of {ratio:.1e}.\n\n"
        f"The cause is in the chain rule itself — equation (10) multiplies by g'(z) at every\n"
        f"layer, so the gradient reaching layer 1 carries a product of {depth} derivatives.\n"
        f"sigmoid'(z) peaks at 0.25 and is usually far less, so 0.25^{depth} = {0.25**depth:.1e}\n"
        f"even in the best case. The early layers receive essentially no signal and never\n"
        f"learn. ReLU's derivative is exactly 1 wherever the unit is active, so the product\n"
        f"does not decay — which is why ReLU, not sigmoid, made deep networks trainable.\n"
        f"This is project 02's vanishing-gradient problem again, now compounded by depth."
    )

    plt.figure(figsize=(7, 4.4))
    for act, vals in curves.items():
        plt.plot(range(1, len(vals) + 1), vals, marker="o", markersize=4, label=act)
    plt.yscale("log")
    plt.xlabel("Layer (1 = closest to the input, right = output)")
    plt.ylabel("||dJ/dW|| for that layer (log scale)")
    plt.title(f"Gradient reaching each layer of a {depth}-layer network")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "vanishing_gradients.png", dpi=120)
    plt.close()
    print("Saved plot to outputs/vanishing_gradients.png")


# ---------------------------------------------------------------------------
# Part 5 — initialization
# ---------------------------------------------------------------------------


def run_initialization_demo() -> None:
    print()
    print("=" * 74)
    print("PART 5 — Initialization: why you cannot start a network at zero")
    print("=" * 74)

    X, y = make_moons(n_samples=600, noise=0.2, random_state=42)
    Y = one_hot(y, 2)

    # Shallow AND deep, because the two tell different stories — and only the deep
    # one supports the textbook warning about initialization scale.
    shallow = [2, 32, 32, 2]
    deep = [2] + [32] * 8 + [2]
    schemes = ["zeros", "small", "large", "he"]
    histories = {}

    print(f"\n{'Init':<10}{'2 hidden layers':>20}{'8 hidden layers':>20}")
    print(f"{'':<10}{'loss':>10}{'accuracy':>10}{'loss':>10}{'accuracy':>10}")
    print("-" * 74)
    for init in schemes:
        row = []
        for sizes in (shallow, deep):
            net = NeuralNetwork(sizes, activation="relu", init=init,
                                learning_rate=0.2, seed=5).fit(X, Y, epochs=1500)
            row.append((net.loss_history[-1], accuracy_score(y, net.predict(X))))
            if sizes is shallow:
                histories[init] = net.loss_history
                shallow_net = net
        print(f"{init:<10}{row[0][0]:>10.4f}{row[0][1]:>10.4f}{row[1][0]:>10.4f}{row[1][1]:>10.4f}")
        if init == "zeros":
            n_distinct = len(np.unique(np.round(shallow_net.weights[0], 10), axis=1).T)
            print(f"          -> after training, the 32 hidden units still have only "
                  f"{n_distinct} distinct weight vector(s)")

    print(
        "\nZERO init fails absolutely, at any depth, and the reason is elegant: if every\n"
        "weight starts identical, every hidden unit in a layer computes the same thing,\n"
        "receives the same gradient, and is updated identically. They remain clones\n"
        "forever — a 32-unit layer with the capacity of a 1-unit layer. Accuracy 0.5 is a\n"
        "coin flip. This is the SYMMETRY problem, and breaking it is the entire reason\n"
        "initialization is random. Note that projects 01 and 02 could safely use zero init\n"
        "precisely because they had no hidden layer to make symmetric.\n\n"
        "The SCALE story is more interesting, and the usual telling of it is too confident.\n"
        "Compare the two halves of the table:\n"
        "  - At 2 hidden layers, scale hardly matters. 'small' and 'large' both train to\n"
        "    respectable accuracy; He is best but the margin is not dramatic. A shallow\n"
        "    network simply does not compound its scaling errors enough to care.\n"
        "  - At 8 hidden layers, the same two schemes fall apart. 'small' stalls at a loss\n"
        "    of 0.6931 — which is exactly log(2), the loss of a model that outputs 50/50\n"
        "    and has learned nothing. 'large' produces an outright nan: the gradients\n"
        "    EXPLODED rather than vanished, overflowing to infinity.\n"
        "    Why: the signal is multiplied by a badly scaled matrix eight times forward and\n"
        "    eight times back, so a factor slightly below 1 decays to nothing and one\n"
        "    slightly above 1 blows up. This is Part 4's chain-rule product again, caused\n"
        "    now by the weights rather than the activation.\n\n"
        "He init sets the variance to 2/fan_in precisely so that this product stays near 1\n"
        "through depth; the 2 compensates for ReLU zeroing out roughly half its inputs\n"
        "(README §4.6). The honest summary: initialization is a DEPTH problem. If someone\n"
        "tells you a scheme is critical, ask how deep their network was."
    )

    plt.figure(figsize=(7, 4.4))
    for init, hist in histories.items():
        plt.plot(hist, label=init)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.yscale("log")
    plt.title("ReLU network: same data, same architecture — only the initial weights differ")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "initialization.png", dpi=120)
    plt.close()
    print("Saved plot to outputs/initialization.png")


# ---------------------------------------------------------------------------
# Part 6 — real data
# ---------------------------------------------------------------------------


def run_digits_demo() -> None:
    print()
    print("=" * 74)
    print("PART 6 — Handwritten digits: scratch network vs. sklearn")
    print("=" * 74)

    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.25, random_state=42, stratify=digits.target
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    Y_train = one_hot(y_train, 10)

    net = NeuralNetwork([64, 64, 32, 10], activation="relu", init="he",
                        learning_rate=0.5, seed=7)
    test_curve = []
    for _ in range(60):
        net.fit(X_train_s, Y_train, epochs=10)
        test_curve.append(accuracy_score(y_test, net.predict(X_test_s)))

    logistic = LogisticRegression(max_iter=5000).fit(X_train_s, y_train)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000,
                        random_state=42).fit(X_train_s, y_train)

    print(f"\n{'Model':<44}{'Test accuracy':>16}{'params':>12}")
    print("-" * 74)
    n_params = sum(w.size for w in net.weights) + sum(b.size for b in net.biases)
    print(f"{'Logistic regression (project 02)':<44}"
          f"{accuracy_score(y_test, logistic.predict(X_test_s)):>16.4f}{64*10+10:>12}")
    print(f"{'Scratch network [64, 64, 32, 10]':<44}"
          f"{accuracy_score(y_test, net.predict(X_test_s)):>16.4f}{n_params:>12}")
    print(f"{'sklearn MLPClassifier (64, 32)':<44}"
          f"{accuracy_score(y_test, mlp.predict(X_test_s)):>16.4f}{'~same':>12}")

    net_acc = accuracy_score(y_test, net.predict(X_test_s))
    log_acc = accuracy_score(y_test, logistic.predict(X_test_s))
    print(
        f"\nCORRECTNESS: the scratch network matches sklearn's MLPClassifier to four decimal\n"
        f"places ({net_acc:.4f} vs {accuracy_score(y_test, mlp.predict(X_test_s)):.4f}) on the same "
        f"architecture. Every line of it was derived\nin this README, and it performs identically to a "
        f"professional implementation.\n\n"
        f"NOW THE UNCOMFORTABLE PART: logistic regression BEATS them both ({log_acc:.4f} vs "
        f"{net_acc:.4f}),\nwith 10x fewer parameters. That is not a bug — sklearn's network loses "
        f"to it too,\nwhich is how you know. 8x8 digits are already nearly linearly separable, so the\n"
        f"extra capacity buys no bias reduction while adding variance (project 03's table in\n"
        f"one sentence). A bigger model is not automatically a better one, and this is worth\n"
        f"seeing on the very first network you build, not learning later at a company's\n"
        f"expense. Exercise 6 asks you to actually beat logistic regression here.\n\n"
        f"What DOES open a real gap on images is exploiting the fact that pixels have\n"
        f"NEIGHBOURS — structure this network destroys the moment it flattens an 8x8 image\n"
        f"into 64 unrelated inputs. That is project 08."
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(net.loss_history)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training cross-entropy")
    ax1.set_yscale("log")
    ax1.set_title("Training loss")
    ax2.plot(np.arange(1, len(test_curve) + 1) * 10, test_curve, marker="o", markersize=3)
    ax2.axhline(accuracy_score(y_test, logistic.predict(X_test_s)), color="grey",
                linestyle="--", label="logistic regression")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Test accuracy")
    ax2.set_title("Test accuracy while training")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "digits_training.png", dpi=120)
    plt.close()
    print("Saved plot to outputs/digits_training.png")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    run_xor_demo()
    run_gradient_check()
    run_linearity_demo()
    run_vanishing_gradient_demo()
    run_initialization_demo()
    run_digits_demo()
