"""
Trees & Ensembles — a decision tree built from scratch on entropy and information
gain, then the two ways of combining many trees, each demonstrated rather than
asserted:

  Part 1  entropy and information gain, with the arithmetic of one real split printed
  Part 2  a single tree overfits — depth sweep, and the axis-aligned boundary it draws
  Part 3  bagging: verifying Var = rho*sigma^2 + (1-rho)*sigma^2/B numerically
  Part 4  random forests: feature subsampling lowers rho, which is the whole trick
  Part 5  gradient boosting from scratch — residuals ARE the negative gradient
  Part 6  bias-variance table: bagging kills variance, boosting kills bias

Run:
    python trees_and_ensembles.py

See README.md for the math behind every formula referenced in the comments below.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write plots to file; don't require a display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.ensemble import (
    BaggingRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

RNG = np.random.default_rng(seed=42)
OUTPUT_DIR = Path(__file__).parent / "outputs"


# ---------------------------------------------------------------------------
# Impurity measures
# ---------------------------------------------------------------------------


def entropy(y: np.ndarray) -> float:
    """
    H(S) = -sum_k p_k * log2(p_k)                                              (1)

    "How many bits does it take to describe the label of a random member of S?"
    A pure node needs 0 bits (you already know the answer). A 50/50 binary split
    needs exactly 1 bit — maximum uncertainty. Terms with p_k = 0 are dropped
    because p*log(p) -> 0 as p -> 0, and log(0) is undefined.
    """
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y)
    probs = counts[counts > 0] / len(y)
    return float(-np.sum(probs * np.log2(probs)))


def gini(y: np.ndarray) -> float:
    """
    Gini(S) = 1 - sum_k p_k^2                                                  (2)

    "Label a random member by drawing a label at random from S's distribution —
    how often are you wrong?" Behaves almost identically to entropy in practice
    (see exercise 2) but avoids a log, which is why it's sklearn's default.
    """
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y)
    probs = counts[counts > 0] / len(y)
    return float(1.0 - np.sum(probs**2))


def information_gain(y_parent: np.ndarray, y_left: np.ndarray, y_right: np.ndarray,
                     impurity=entropy) -> float:
    """
    IG = H(parent) - [ (n_L/n) H(left) + (n_R/n) H(right) ]                    (3)

    The impurity you had, minus the impurity you're left with — where "left with"
    is the *weighted* average over children, weighted by how many samples land in
    each. The weighting is essential: a split that isolates 2 pure samples out of
    500 has barely reduced anyone's uncertainty.
    """
    n = len(y_parent)
    if n == 0 or len(y_left) == 0 or len(y_right) == 0:
        return 0.0
    weighted_child = (len(y_left) / n) * impurity(y_left) + (len(y_right) / n) * impurity(y_right)
    return impurity(y_parent) - weighted_child


# ---------------------------------------------------------------------------
# A decision tree, from scratch
# ---------------------------------------------------------------------------


class Node:
    """Either a decision (feature, threshold, two children) or a leaf (a prediction)."""

    def __init__(self, *, feature=None, threshold=None, left=None, right=None, prediction=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction

    @property
    def is_leaf(self) -> bool:
        return self.prediction is not None


class DecisionTreeScratch:
    """
    Greedy recursive binary splitting (the CART algorithm):

        at each node, try every (feature, threshold) pair, score each by
        information gain (3), take the best, recurse on both sides.

    "Greedy" means it never reconsiders: it takes the locally best split and moves
    on. Finding the globally optimal tree is NP-hard, so every practical tree
    implementation — sklearn's included — is greedy. See README §3.3.

    Stopping rules (max_depth, min_samples_split) exist because an unconstrained
    tree will happily split until every leaf holds one sample, which is a perfect
    memorization of the training data. Part 2 shows that happening.
    """

    def __init__(self, max_depth: int | None = None, min_samples_split: int = 2,
                 criterion: str = "entropy", max_candidate_thresholds: int = 32):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.impurity = entropy if criterion == "entropy" else gini
        self.max_candidate_thresholds = max_candidate_thresholds
        self.root: Node | None = None

    def _candidate_thresholds(self, column: np.ndarray) -> np.ndarray:
        """
        Split points to try. Only midpoints *between* distinct observed values can
        separate samples differently, so those are the only candidates worth
        scoring. On continuous features that's still up to n-1 per feature, so we
        subsample to a fixed number of quantiles to keep this from being O(n) per
        feature per node — sklearn does the same thing when given many samples.
        """
        values = np.unique(column)
        if len(values) <= 1:
            return np.array([])
        if len(values) > self.max_candidate_thresholds:
            qs = np.linspace(0, 1, self.max_candidate_thresholds + 2)[1:-1]
            values = np.unique(np.quantile(values, qs))
        return (values[:-1] + values[1:]) / 2.0

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        best_gain, best_feature, best_threshold = 0.0, None, None
        for feature in range(X.shape[1]):
            column = X[:, feature]
            for threshold in self._candidate_thresholds(column):
                mask = column <= threshold
                gain = information_gain(y, y[mask], y[~mask], self.impurity)
                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, feature, threshold
        return best_feature, best_threshold, best_gain

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        # Stop if pure, too small, or too deep — then predict the majority label.
        if (
            len(np.unique(y)) == 1
            or len(y) < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return Node(prediction=int(np.bincount(y).argmax()))

        feature, threshold, gain = self._best_split(X, y)
        if feature is None or gain <= 0:  # no split improves anything
            return Node(prediction=int(np.bincount(y).argmax()))

        mask = X[:, feature] <= threshold
        return Node(
            feature=feature,
            threshold=threshold,
            left=self._build(X[mask], y[mask], depth + 1),
            right=self._build(X[~mask], y[~mask], depth + 1),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeScratch":
        self.root = self._build(X, y.astype(int), depth=0)
        return self

    def _predict_one(self, x: np.ndarray) -> int:
        node = self.root
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.prediction

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(x) for x in X])


# ---------------------------------------------------------------------------
# Part 1 — entropy and information gain, with real arithmetic
# ---------------------------------------------------------------------------


def run_entropy_demo() -> None:
    print("=" * 74)
    print("PART 1 — Entropy, information gain, and a tree built from them")
    print("=" * 74)

    # A tiny hand-checkable example first.
    pure = np.array([1, 1, 1, 1])
    even = np.array([0, 0, 1, 1])
    skewed = np.array([0, 0, 0, 1])
    print("\nEntropy is 'bits of uncertainty about the label':")
    for name, arr in [("[1,1,1,1]  (pure)", pure), ("[0,0,1,1]  (50/50)", even),
                      ("[0,0,0,1]  (75/25)", skewed)]:
        print(f"  H({name}) = {entropy(arr):.4f} bits     Gini = {gini(arr):.4f}")

    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Show the arithmetic of the actual root split the tree chooses.
    tree = DecisionTreeScratch(max_depth=4)
    feature, threshold, gain = tree._best_split(X_train, y_train)
    mask = X_train[:, feature] <= threshold
    y_left, y_right = y_train[mask], y_train[~mask]
    n, n_l, n_r = len(y_train), len(y_left), len(y_right)

    print(f"\nBest root split found: '{data.feature_names[feature]}' <= {threshold:.4f}")
    print(f"  H(parent) = {entropy(y_train):.4f}   over {n} samples "
          f"({np.bincount(y_train)[1]} benign / {np.bincount(y_train)[0]} malignant)")
    print(f"  H(left)   = {entropy(y_left):.4f}   over {n_l} samples "
          f"({np.bincount(y_left, minlength=2)[1]} benign / {np.bincount(y_left, minlength=2)[0]} malignant)")
    print(f"  H(right)  = {entropy(y_right):.4f}   over {n_r} samples "
          f"({np.bincount(y_right, minlength=2)[1]} benign / {np.bincount(y_right, minlength=2)[0]} malignant)")
    print(f"\n  IG = {entropy(y_train):.4f} - [ ({n_l}/{n})*{entropy(y_left):.4f} "
          f"+ ({n_r}/{n})*{entropy(y_right):.4f} ]")
    print(f"     = {entropy(y_train):.4f} - {(n_l/n)*entropy(y_left) + (n_r/n)*entropy(y_right):.4f} "
          f"= {gain:.4f} bits")

    tree.fit(X_train, y_train)
    sk = DecisionTreeClassifier(max_depth=4, criterion="entropy", random_state=42)
    sk.fit(X_train, y_train)
    print(f"\n{'Method (max_depth=4, entropy)':<38}{'Test accuracy':>14}")
    print("-" * 74)
    print(f"{'Scratch tree':<38}{accuracy_score(y_test, tree.predict(X_test)):>14.4f}")
    print(f"{'sklearn DecisionTreeClassifier':<38}{accuracy_score(y_test, sk.predict(X_test)):>14.4f}")
    print(
        "\nClose but not necessarily identical: sklearn considers every midpoint as a\n"
        "candidate threshold while this implementation subsamples to 32 quantiles per\n"
        "feature, and ties are broken differently. Same algorithm, different bookkeeping."
    )


# ---------------------------------------------------------------------------
# Part 2 — one tree overfits
# ---------------------------------------------------------------------------


def run_overfitting_demo() -> None:
    print()
    print("=" * 74)
    print("PART 2 — A single tree memorizes: depth vs. train/test accuracy")
    print("=" * 74)

    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3, random_state=42, stratify=data.target
    )

    depths = list(range(1, 21))
    train_acc, test_acc, n_leaves = [], [], []
    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, random_state=42).fit(X_train, y_train)
        train_acc.append(accuracy_score(y_train, clf.predict(X_train)))
        test_acc.append(accuracy_score(y_test, clf.predict(X_test)))
        n_leaves.append(clf.get_n_leaves())

    best = depths[int(np.argmax(test_acc))]
    full = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
    print(f"\n{'depth':>6}{'train acc':>12}{'test acc':>11}{'leaves':>9}")
    print("-" * 74)
    for d, tr, te, nl in zip(depths, train_acc, test_acc, n_leaves):
        if d <= 8 or d % 4 == 0:
            print(f"{d:>6}{tr:>12.4f}{te:>11.4f}{nl:>9}")
    print(
        f"\nTraining accuracy reaches {max(train_acc):.4f} — a fully grown tree splits until every\n"
        f"leaf is pure, which is memorization, not learning. Test accuracy peaks at depth "
        f"{best}\n({max(test_acc):.4f}) and then flattens/declines. An unrestricted tree grows "
        f"{full.get_n_leaves()} leaves for\n{len(y_train)} training samples. This is project 03's "
        f"U-curve again, with depth as the\nflexibility knob instead of polynomial degree."
    )

    # 2D boundary: what a tree's decision surface actually looks like.
    Xm, ym = make_moons(n_samples=400, noise=0.3, random_state=42)
    xx, yy = np.meshgrid(
        np.linspace(Xm[:, 0].min() - 0.5, Xm[:, 0].max() + 0.5, 300),
        np.linspace(Xm[:, 1].min() - 0.5, Xm[:, 1].max() + 0.5, 300),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (title, model) in zip(axes, [
        ("Tree, max_depth=3 (underfit)", DecisionTreeClassifier(max_depth=3, random_state=42)),
        ("Tree, unrestricted (overfit)", DecisionTreeClassifier(random_state=42)),
        ("Random forest, 300 trees", RandomForestClassifier(n_estimators=300, random_state=42)),
    ]):
        model.fit(Xm, ym)
        ax.contourf(xx, yy, model.predict(grid).reshape(xx.shape), alpha=0.25, cmap="coolwarm")
        ax.scatter(Xm[:, 0], Xm[:, 1], c=ym, cmap="coolwarm", s=14, edgecolor="k", linewidth=0.3)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tree_boundaries.png", dpi=120)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(depths, train_acc, marker="o", label="Training accuracy")
    plt.plot(depths, test_acc, marker="s", label="Test accuracy")
    plt.axvline(best, color="black", linestyle="--", linewidth=1, label=f"best depth ({best})")
    plt.xlabel("max_depth (model flexibility)")
    plt.ylabel("Accuracy")
    plt.title("A single tree: training accuracy hits 1.0 and keeps going")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tree_depth_overfitting.png", dpi=120)
    plt.close()
    print("Saved plots to outputs/tree_depth_overfitting.png and tree_boundaries.png")


# ---------------------------------------------------------------------------
# Parts 3 & 4 — bagging, random forests, and the variance formula
# ---------------------------------------------------------------------------


FRIEDMAN_NOISE = 1.0


def friedman_f(X: np.ndarray) -> np.ndarray:
    """The true function behind make_friedman1 — known exactly, so bias is computable."""
    return (
        10 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20 * (X[:, 2] - 0.5) ** 2
        + 10 * X[:, 3]
        + 5 * X[:, 4]
    )


def sample_friedman(n: int, rng: np.random.Generator, n_features: int = 10):
    X = rng.uniform(0, 1, size=(n, n_features))
    y = friedman_f(X) + rng.normal(0, FRIEDMAN_NOISE, size=n)
    return X, y


def _measure_sigma_rho(preds: np.ndarray) -> tuple[float, float]:
    """
    From preds[r, b, i] (dataset draw r, tree b, test point i), estimate:
        sigma^2 — variance of ONE tree's prediction across dataset draws
        rho     — correlation between two DIFFERENT trees in the same ensemble
    """
    sigma_sq = float(preds.var(axis=0).mean())
    centred = preds - preds.mean(axis=0, keepdims=True)
    n_repeats, n_trees, n_points = preds.shape
    per_point = []
    for i in range(n_points):
        C = (centred[:, :, i].T @ centred[:, :, i]) / n_repeats  # (n_trees, n_trees)
        per_point.append((C.sum() - np.trace(C)) / (n_trees * (n_trees - 1)))
    return sigma_sq, float(np.mean(per_point)) / sigma_sq


def _ensemble_variance_study(max_features, label: str, n_repeats=40, n_trees=100,
                             n_fit=25, n_train=200):
    """
    A genuine out-of-sample test of the variance formula (README §3.5).

    IMPORTANT — why this is set up the way it is: given sigma^2 and rho measured on
    the SAME B trees you then predict, the formula is an algebraic identity, so
    "predicted == measured" would be arithmetic, not evidence. To make it a real
    prediction, we estimate sigma^2 and rho from the first `n_fit` trees only, use
    them to predict the variance of a much larger ensemble of `n_trees`, and then
    measure that larger ensemble directly. Nothing forces those to agree.
    """
    rng = np.random.default_rng(11)
    X_test, _ = sample_friedman(150, rng)

    # preds[r, b, i] = prediction of tree b (trained on dataset r) at test point i
    preds = np.empty((n_repeats, n_trees, len(X_test)))
    for r in range(n_repeats):
        X_tr, y_tr = sample_friedman(n_train, rng)
        for b in range(n_trees):
            boot = rng.integers(0, n_train, n_train)  # bootstrap resample
            tree = DecisionTreeRegressor(
                max_features=max_features, random_state=int(rng.integers(0, 10**6))
            )
            tree.fit(X_tr[boot], y_tr[boot])
            preds[r, b] = tree.predict(X_test)

    sigma_sq, rho = _measure_sigma_rho(preds[:, :n_fit, :])  # fit on 25 trees...
    predicted = rho * sigma_sq + (1 - rho) * sigma_sq / n_trees  # ...predict for 100...
    measured = float(preds.mean(axis=1).var(axis=0).mean())  # ...and check against 100
    return {
        "label": label, "sigma_sq": sigma_sq, "rho": rho, "predicted": predicted,
        "measured": measured, "n_trees": n_trees, "n_fit": n_fit,
        "floor": rho * sigma_sq,
    }


def run_bagging_demo() -> None:
    print()
    print("=" * 74)
    print("PART 3 & 4 — Bagging, random forests, and why decorrelation is the trick")
    print("=" * 74)

    bagging = _ensemble_variance_study(max_features=None, label="Bagging (all 10 features)")
    forest = _ensemble_variance_study(max_features=0.3, label="Random forest (3 of 10)")

    print(f"\nsigma^2 and rho estimated from {bagging['n_fit']} trees; used to predict the variance")
    print(f"of a {bagging['n_trees']}-tree ensemble, which is then measured directly.\n")
    print(f"{'Ensemble':<30}{'sigma^2':>10}{'rho':>8}{'floor':>9}"
          f"{'predicted':>11}{'measured':>10}")
    print("-" * 74)
    for r in (bagging, forest):
        print(f"{r['label']:<30}{r['sigma_sq']:>10.3f}{r['rho']:>8.3f}{r['floor']:>9.3f}"
              f"{r['predicted']:>11.3f}{r['measured']:>10.3f}")

    err = max(abs(r["predicted"] - r["measured"]) for r in (bagging, forest))
    print(
        f"\nThe prediction is out-of-sample — rho and sigma^2 come from {bagging['n_fit']} trees, the\n"
        f"measurement from {bagging['n_trees']} — and it lands within {err:.3f}. Read what the formula says:\n"
        f"  - The (1-rho)*sigma^2/B term vanishes as B grows. Adding trees is free variance\n"
        f"    reduction, and unlike most knobs in ML it cannot overfit — more is never worse.\n"
        f"  - But the FIRST term, rho*sigma^2, has no B in it at all. It is a hard floor.\n"
        f"    With rho = {bagging['rho']:.3f}, bagging can never push variance below "
        f"{bagging['floor']:.3f},\n"
        f"    no matter how many trees you add. Averaging cannot remove shared error.\n"
        f"  - Random forests attack rho itself: letting each split see only 3 of the 10\n"
        f"    features forces the trees to disagree. rho falls {bagging['rho']:.3f} -> "
        f"{forest['rho']:.3f} and the floor falls\n"
        f"    {bagging['floor']:.3f} -> {forest['floor']:.3f}. That decorrelation IS the random forest.\n"
        f"  - Note sigma^2 goes UP ({bagging['sigma_sq']:.1f} -> {forest['sigma_sq']:.1f}): each individual "
        f"tree is worse, because\n    it is sometimes denied the feature it wanted. Part 6 shows that this "
        f"trade is\n    not always worth it."
    )

    # Show the floor directly: sweep B, and mark the two measured points.
    Bs = np.array([1, 2, 3, 5, 8, 12, 20, 35, 60, 100])
    plt.figure(figsize=(7, 4.2))
    for r in (bagging, forest):
        line, = plt.plot(
            Bs, r["rho"] * r["sigma_sq"] + (1 - r["rho"]) * r["sigma_sq"] / Bs,
            marker="o", label=f"{r['label']} — rho={r['rho']:.3f}",
        )
        plt.axhline(r["floor"], color=line.get_color(), linestyle=":", linewidth=1)
    plt.scatter([bagging["n_trees"], forest["n_trees"]],
                [bagging["measured"], forest["measured"]],
                color="black", zorder=5, marker="X", s=90, label="measured at B=100")
    plt.xscale("log")
    plt.xlabel("Number of trees B (log scale)")
    plt.ylabel("Variance of the ensemble's prediction")
    plt.title("Averaging removes the (1-rho)/B term; only lowering rho moves the floor")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bagging_variance.png", dpi=120)
    plt.close()
    print("Saved plot to outputs/bagging_variance.png (dotted lines = the rho*sigma^2 floor)")


# ---------------------------------------------------------------------------
# Part 5 — gradient boosting from scratch
# ---------------------------------------------------------------------------


class GradientBoostingScratch:
    """
    Boosting for squared-error regression:                                     (4)

        F_0(x)   = mean(y)
        r_m      = y - F_{m-1}(x)          <- residual = NEGATIVE GRADIENT
        h_m      = a shallow tree fit to r_m
        F_m(x)   = F_{m-1}(x) + lr * h_m(x)

    The key identity, derived in README §3.6: for L = (1/2)(y - F)^2,

        -dL/dF = y - F = the residual

    so "fit the next tree to the residuals" is not a heuristic — it is gradient
    descent, taking steps in the space of *functions* rather than the space of
    parameters. lr is the learning rate, exactly as in projects 01 and 02.
    """

    def __init__(self, n_estimators: int = 200, learning_rate: float = 0.05, max_depth: int = 2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees: list[DecisionTreeRegressor] = []
        self.initial_prediction = 0.0
        self.train_loss_history: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingScratch":
        self.initial_prediction = float(np.mean(y))  # F_0
        current = np.full(len(y), self.initial_prediction)
        self.trees, self.train_loss_history = [], []

        for _ in range(self.n_estimators):
            residual = y - current  # the negative gradient
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=0)
            tree.fit(X, residual)
            current += self.learning_rate * tree.predict(X)
            self.trees.append(tree)
            self.train_loss_history.append(float(np.mean((y - current) ** 2)))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            out += self.learning_rate * tree.predict(X)
        return out


def run_boosting_demo() -> None:
    print()
    print("=" * 74)
    print("PART 5 — Gradient boosting from scratch: the residual IS the gradient")
    print("=" * 74)

    rng = np.random.default_rng(3)
    X_train, y_train = sample_friedman(400, rng)
    X_test, y_test = sample_friedman(400, rng)

    model = GradientBoostingScratch(n_estimators=300, learning_rate=0.05, max_depth=2)
    model.fit(X_train, y_train)

    sk = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=2, random_state=0
    ).fit(X_train, y_train)
    stump = DecisionTreeRegressor(max_depth=2, random_state=0).fit(X_train, y_train)

    print(f"\n{'Model':<44}{'Test MSE':>11}")
    print("-" * 74)
    print(f"{'A single depth-2 tree (the weak learner)':<44}"
          f"{mean_squared_error(y_test, stump.predict(X_test)):>11.4f}")
    print(f"{'Scratch gradient boosting (300 x depth-2)':<44}"
          f"{mean_squared_error(y_test, model.predict(X_test)):>11.4f}")
    print(f"{'sklearn GradientBoostingRegressor':<44}"
          f"{mean_squared_error(y_test, sk.predict(X_test)):>11.4f}")
    print(f"{'Irreducible noise floor (sigma^2)':<44}{FRIEDMAN_NOISE**2:>11.4f}")

    print(
        f"\n300 stumps that individually score {mean_squared_error(y_test, stump.predict(X_test)):.2f} "
        f"combine into a model scoring\n"
        f"{mean_squared_error(y_test, model.predict(X_test)):.2f} — closing most of the distance to the "
        f"noise floor of {FRIEDMAN_NOISE**2:.2f}. Each tree is fit\nto what the previous ones got wrong, "
        f"so the ensemble corrects its own bias.\nThe scratch and sklearn numbers agree closely; both "
        f"implement the same update (4)."
    )

    # Track test error alongside train error to show boosting CAN overfit eventually.
    test_curve = []
    running = np.full(len(y_test), model.initial_prediction)
    for tree in model.trees:
        running += model.learning_rate * tree.predict(X_test)
        test_curve.append(mean_squared_error(y_test, running))

    plt.figure(figsize=(7, 4.2))
    plt.plot(model.train_loss_history, label="Training MSE")
    plt.plot(test_curve, label="Test MSE")
    plt.axhline(FRIEDMAN_NOISE**2, color="grey", linestyle=":", label="irreducible noise (sigma^2)")
    plt.xlabel("Number of boosting rounds")
    plt.ylabel("Mean squared error")
    plt.yscale("log")
    plt.title("Boosting: each tree fits the previous ensemble's residuals")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "boosting_curve.png", dpi=120)
    plt.close()
    print("Saved plot to outputs/boosting_curve.png")


# ---------------------------------------------------------------------------
# Part 6 — the payoff: which error term does each ensemble attack?
# ---------------------------------------------------------------------------


def run_bias_variance_table() -> None:
    print()
    print("=" * 74)
    print("PART 6 — Project 03's decomposition applied: what does each ensemble fix?")
    print("=" * 74)

    n_datasets, n_train = 60, 200
    rng = np.random.default_rng(5)
    X_test, _ = sample_friedman(200, rng)
    f_true = friedman_f(X_test)

    models = {
        "Single tree (unrestricted)": lambda: DecisionTreeRegressor(random_state=0),
        "Single tree (max_depth=3)": lambda: DecisionTreeRegressor(max_depth=3, random_state=0),
        "Bagging (50 trees)": lambda: BaggingRegressor(
            DecisionTreeRegressor(random_state=0), n_estimators=50, random_state=0),
        "Random forest (50, 3 feats)": lambda: RandomForestRegressor(
            n_estimators=50, max_features=0.3, random_state=0),
        "Boosting (300 x depth-2)": lambda: GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=2, random_state=0),
    }

    datasets = [sample_friedman(n_train, rng) for _ in range(n_datasets)]
    print(f"\n{'Model':<30}{'bias²':>10}{'variance':>11}{'noise':>9}{'total':>10}")
    print("-" * 74)
    for name, make in models.items():
        preds = np.empty((n_datasets, len(X_test)))
        for d, (X_tr, y_tr) in enumerate(datasets):
            preds[d] = make().fit(X_tr, y_tr).predict(X_test)
        bias_sq = np.mean((preds.mean(axis=0) - f_true) ** 2)
        variance = np.mean(preds.var(axis=0))
        total = bias_sq + variance + FRIEDMAN_NOISE**2
        print(f"{name:<30}{bias_sq:>10.3f}{variance:>11.3f}{FRIEDMAN_NOISE**2:>9.3f}{total:>10.3f}")

    print(
        "\nRead down the columns — this is the whole chapter in one table:\n"
        "  - The unrestricted tree has the lowest bias of any single tree and by far the\n"
        "    largest variance. It CAN fit the truth, but WHICH truth it fits depends\n"
        "    heavily on the sample it happened to see.\n"
        "  - Capping depth at 3 trades that the other way: variance nearly halves, bias\n"
        "    almost doubles. Same model family, opposite failure mode.\n"
        "  - Bagging keeps the deep tree's low bias while cutting variance ~6x. That is\n"
        "    the point of averaging: it does not change the expected model, only the\n"
        "    spread around it.\n"
        "  - The random forest cuts variance further still (lower rho, lower floor) — but\n"
        "    on THIS dataset its bias rises enough that its total error is WORSE than\n"
        "    plain bagging. Only 5 of the 10 features carry signal, so restricting each\n"
        "    split to 3 random features often hides all of them. Decorrelation is not\n"
        "    free, and max_features is a hyperparameter to tune, not a free win.\n"
        "    (Exercise 4 sweeps it and finds where the trade turns favourable.)\n"
        "  - Boosting attacks the other term: it stacks HIGH-bias depth-2 stumps and\n"
        "    drives bias down by fitting residuals, ending with the lowest total error\n"
        "    here despite each member being far too weak to use alone.\n"
        "Bagging is a variance fix. Boosting is a bias fix. Knowing which term dominates\n"
        "your error — measured, as in project 03 — is how you choose between them."
    )


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    run_entropy_demo()
    run_overfitting_demo()
    run_bagging_demo()
    run_boosting_demo()
    run_bias_variance_table()
