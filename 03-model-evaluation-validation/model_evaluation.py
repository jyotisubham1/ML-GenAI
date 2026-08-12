"""
Model Evaluation & Validation — the metrics themselves implemented from scratch,
then five experiments that each demonstrate a claim about evaluation instead of
asserting it:

  Part 1  accuracy hides failure on imbalanced data (and what to use instead)
  Part 2  a single train/test split is a noisy estimate; k-fold shrinks that noise
  Part 3  the bias-variance decomposition, verified numerically (bias² + var + noise
          really does add up to the observed error)
  Part 4  ROC-AUC vs. precision-recall AUC, plus a Monte-Carlo proof of what AUC
          actually *means* probabilistically
  Part 5  data leakage manufacturing 75%+ accuracy out of pure random noise

Run:
    python model_evaluation.py

See README.md for the math behind every formula referenced in the comments below.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write plots to file; don't require a display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(seed=42)
OUTPUT_DIR = Path(__file__).parent / "outputs"


# ---------------------------------------------------------------------------
# Metrics from scratch
#
# Every one of these is three lines of counting. The point of writing them out is
# that "precision" and "recall" stop being names you look up and become obvious
# consequences of which cell of the confusion matrix you divide by.
# ---------------------------------------------------------------------------


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    """
    Returns (TN, FP, FN, TP).                                                  (1)

    The whole of binary classification evaluation is these four numbers. Every
    metric below is a ratio of two of them — the only question each metric asks is
    "out of *what* total?".
    """
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))  # predicted positive, wasn't
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))  # missed a real positive
    return tn, fp, fn, tp


def accuracy_scratch(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """accuracy = (TP + TN) / (TP + TN + FP + FN)                              (2)"""
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    return (tp + tn) / (tp + tn + fp + fn)


def precision_scratch(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    precision = TP / (TP + FP)                                                 (3)

    Denominator = everything the model *called* positive. "When it raises an alarm,
    how often is it right?" Undefined if the model never predicts positive — we
    return 0.0, which is also sklearn's zero_division default.
    """
    _, fp, _, tp = confusion_counts(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall_scratch(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    recall = TP / (TP + FN)                                                    (4)

    Denominator = everything that *actually is* positive. "Of the real cases out
    there, how many did it catch?" Note the denominator does not depend on the
    model at all — it's fixed by the data. That's what makes recall impossible to
    game by predicting positive less often.
    """
    _, _, fn, tp = confusion_counts(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_scratch(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    F1 = 2 * (precision * recall) / (precision + recall)                       (5)

    The *harmonic* mean, not the arithmetic one. That matters: the harmonic mean is
    dominated by the smaller of the two. precision=1.0, recall=0.0 gives arithmetic
    mean 0.5 (looks mediocre) but F1 = 0.0 (correctly: useless). See README §3.
    """
    p = precision_scratch(y_true, y_pred)
    r = recall_scratch(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def roc_curve_scratch(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    ROC curve: (FPR, TPR) as the decision threshold sweeps from +inf down to -inf. (6)

        TPR = TP / (TP + FN) = recall           (y-axis)
        FPR = FP / (FP + TN)                    (x-axis)

    Rather than looping over candidate thresholds, sort by score descending and walk
    down the list: accepting the top k scores as "positive" IS the classifier at the
    threshold equal to the k-th score. Cumulative sums then give TP and FP at every
    threshold at once.
    """
    order = np.argsort(-scores, kind="mergesort")  # stable, so ties keep input order
    y = y_true[order]
    s = scores[order]

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    tps = np.cumsum(y)  # true positives among the top-k
    fps = np.cumsum(1 - y)  # false positives among the top-k

    # Only keep the last index of each run of tied scores: a threshold cannot split
    # points that share a score, so intermediate positions aren't reachable classifiers.
    last_of_tie = np.r_[np.where(np.diff(s))[0], len(y) - 1]

    tpr = np.r_[0.0, tps[last_of_tie] / n_pos]  # prepend the "predict nothing" corner
    fpr = np.r_[0.0, fps[last_of_tie] / n_neg]
    return fpr, tpr


def pr_curve_scratch(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Precision-recall curve, same sweep as (6) but plotting precision against recall. (7)

    Note precision's denominator (TP + FP = k, the number of points accepted) grows
    as the threshold drops, which is why this curve is not monotonic the way ROC is.
    """
    order = np.argsort(-scores, kind="mergesort")
    y = y_true[order]
    s = scores[order]

    n_pos = y.sum()
    tps = np.cumsum(y)
    k = np.arange(1, len(y) + 1)  # how many we've accepted as positive

    last_of_tie = np.r_[np.where(np.diff(s))[0], len(y) - 1]
    recall = tps[last_of_tie] / n_pos
    precision = tps[last_of_tie] / k[last_of_tie]
    return recall, precision


def auc_trapezoid(x: np.ndarray, y: np.ndarray) -> float:
    """Area under a curve by the trapezoid rule — ∫y dx approximated segment by segment. (8)"""
    return float(np.trapezoid(y, x))


def average_precision_scratch(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Average precision — the *step-wise* area under the PR curve:               (8b)

        AP = Σ_n (R_n − R_{n−1}) · P_n

    Not the same number as the trapezoid rule applied to the same curve. Trapezoid
    linearly interpolates between adjacent operating points, and on a PR curve that
    interpolation is not achievable by any real threshold — it optimistically
    "connects the dots" through classifiers that don't exist. AP takes the precision
    actually attained at each point instead, so it's the honest summary and it's what
    sklearn's average_precision_score reports. See README §6.
    """
    recall, precision = pr_curve_scratch(y_true, scores)
    r_prev, ap = 0.0, 0.0
    for r, p in zip(recall, precision):
        ap += (r - r_prev) * p
        r_prev = r
    return float(ap)


def kfold_indices(n_samples: int, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Split 0..n-1 into k contiguous folds; yield (train_idx, val_idx) k times.    (9)

    Each sample lands in the validation set exactly once, so every sample is scored
    by a model that never saw it — that's the property that makes CV an honest
    estimate. (No shuffling here on purpose: callers shuffle first, so the shuffle
    is visible at the call site rather than hidden in here.)
    """
    fold_sizes = np.full(n_splits, n_samples // n_splits)
    fold_sizes[: n_samples % n_splits] += 1  # spread the remainder over the first folds

    indices = np.arange(n_samples)
    splits, start = [], 0
    for size in fold_sizes:
        val_idx = indices[start : start + size]
        train_idx = np.concatenate([indices[:start], indices[start + size :]])
        splits.append((train_idx, val_idx))
        start += size
    return splits


# ---------------------------------------------------------------------------
# Part 1 — accuracy is the wrong headline number on imbalanced data
# ---------------------------------------------------------------------------


def make_imbalanced(n_samples: int = 4000):
    """A deliberately imbalanced binary problem: ~3% positives, like real fraud/disease data."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=12,
        n_informative=4,
        n_redundant=2,
        weights=[0.97, 0.03],
        flip_y=0.01,
        class_sep=0.9,
        random_state=42,
    )
    return X, y


def run_accuracy_paradox_demo() -> None:
    print("=" * 74)
    print("PART 1 — Accuracy hides failure: the same 'good' score, two useless models")
    print("=" * 74)

    X, y = make_imbalanced()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    prevalence = y_test.mean()
    print(f"Positives in the test set: {y_test.sum()} / {len(y_test)}  ({prevalence:.1%})")

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # The "model" that does nothing at all: always predict the majority class.
    y_majority = np.zeros_like(y_test)

    print(f"\n{'Model':<34}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}")
    print("-" * 74)
    for name, pred in [
        ("Always predict majority (no model)", y_majority),
        ("Logistic regression", y_pred),
    ]:
        print(
            f"{name:<34}{accuracy_scratch(y_test, pred):>10.3f}"
            f"{precision_scratch(y_test, pred):>11.3f}"
            f"{recall_scratch(y_test, pred):>9.3f}{f1_scratch(y_test, pred):>8.3f}"
        )

    print(
        f"\nThe do-nothing baseline scores {accuracy_scratch(y_test, y_majority):.1%} accuracy while "
        f"catching zero\npositives. Accuracy's denominator is dominated by the majority class, so on "
        f"a\n{1 - prevalence:.0%}/{prevalence:.0%} split it mostly measures the class balance, not the model."
    )

    # Confirm the from-scratch metrics agree with sklearn — if these disagree,
    # every number above is suspect.
    tn, fp, fn, tp = confusion_counts(y_test, y_pred)
    sk_tn, sk_fp, sk_fn, sk_tp = confusion_matrix(y_test, y_pred).ravel()
    checks = {
        "confusion matrix": (tn, fp, fn, tp) == (sk_tn, sk_fp, sk_fn, sk_tp),
        "accuracy": np.isclose(accuracy_scratch(y_test, y_pred), accuracy_score(y_test, y_pred)),
        "precision": np.isclose(precision_scratch(y_test, y_pred), precision_score(y_test, y_pred)),
        "recall": np.isclose(recall_scratch(y_test, y_pred), recall_score(y_test, y_pred)),
        "f1": np.isclose(f1_scratch(y_test, y_pred), f1_score(y_test, y_pred)),
    }
    print(f"\nConfusion matrix (scratch): TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print("Scratch vs. sklearn agreement: " + ", ".join(f"{k}={'ok' if v else 'MISMATCH'}" for k, v in checks.items()))

    # The threshold is a free parameter — it is NOT part of the fitted model.
    scores = model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0.01, 0.99, 99)
    prec = [precision_scratch(y_test, (scores >= t).astype(int)) for t in thresholds]
    rec = [recall_scratch(y_test, (scores >= t).astype(int)) for t in thresholds]
    f1s = [f1_scratch(y_test, (scores >= t).astype(int)) for t in thresholds]
    best_t = thresholds[int(np.argmax(f1s))]

    plt.figure(figsize=(7, 4))
    plt.plot(thresholds, prec, label="Precision")
    plt.plot(thresholds, rec, label="Recall")
    plt.plot(thresholds, f1s, label="F1", linestyle="--")
    plt.axvline(0.5, color="grey", linewidth=1, label="default threshold (0.5)")
    plt.axvline(best_t, color="black", linewidth=1, linestyle=":", label=f"best F1 (t={best_t:.2f})")
    plt.xlabel("Decision threshold")
    plt.ylabel("Score")
    plt.title("One trained model, many classifiers — the threshold is yours to choose")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "threshold_tradeoff.png", dpi=120)
    plt.close()

    print(
        f"\nSweeping the threshold on the SAME trained model: F1 peaks at t={best_t:.2f} "
        f"({max(f1s):.3f}),\nnot at the default 0.5 ({f1_scratch(y_test, y_pred):.3f}). The 0.5 cutoff is a "
        f"convention, not a\nproperty of the model — precision and recall move in opposite directions "
        f"along it.\nSaved plot to outputs/threshold_tradeoff.png"
    )


# ---------------------------------------------------------------------------
# Part 2 — one split is a noisy measurement; k-fold averages the noise down
# ---------------------------------------------------------------------------


def run_split_variance_demo() -> None:
    print()
    print("=" * 74)
    print("PART 2 — Why one train/test split can lie to you")
    print("=" * 74)

    data = load_breast_cancer()
    X, y = data.data, data.target
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))

    # Exact same data, exact same model, exact same code — only the random split
    # differs. Any spread we see here is pure measurement noise.
    n_repeats = 200
    single_split_scores = []
    for seed in range(n_repeats):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        model.fit(X_tr, y_tr)
        single_split_scores.append(accuracy_scratch(y_te, model.predict(X_te)))
    single_split_scores = np.array(single_split_scores)

    # k-fold: each *estimate* is now the mean of 5 held-out folds instead of 1.
    cv_scores = []
    for seed in range(n_repeats):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        cv_scores.append(cross_val_score(model, X, y, cv=skf, scoring="accuracy").mean())
    cv_scores = np.array(cv_scores)

    print(f"\n{'Estimator of test accuracy':<30}{'mean':>9}{'std':>9}{'min':>9}{'max':>9}{'spread':>9}")
    print("-" * 74)
    for name, s in [("Single 80/20 split", single_split_scores), ("5-fold CV mean", cv_scores)]:
        print(f"{name:<30}{s.mean():>9.4f}{s.std():>9.4f}{s.min():>9.4f}{s.max():>9.4f}{s.max() - s.min():>9.4f}")

    print(
        f"\nBoth estimate the same quantity and agree on the mean. But a single split's "
        f"answer\nranged over {single_split_scores.max() - single_split_scores.min():.1%} across "
        f"{n_repeats} runs — pick a lucky seed and report "
        f"{single_split_scores.max():.1%},\npick an unlucky one and report "
        f"{single_split_scores.min():.1%}, with nothing else changed. 5-fold CV cuts the\n"
        f"standard deviation by {single_split_scores.std() / cv_scores.std():.1f}x because averaging k "
        f"held-out folds averages down the noise."
    )

    # Verify the from-scratch k-fold splitter matches sklearn's, then check that
    # a hand-rolled CV loop reproduces cross_val_score.
    shuffled = RNG.permutation(len(y))
    Xs, ys = X[shuffled], y[shuffled]
    manual = []
    for train_idx, val_idx in kfold_indices(len(ys), n_splits=5):
        model.fit(Xs[train_idx], ys[train_idx])
        manual.append(accuracy_scratch(ys[val_idx], model.predict(Xs[val_idx])))
    sklearn_equiv = cross_val_score(model, Xs, ys, cv=KFold(n_splits=5), scoring="accuracy")
    print(
        f"\nScratch k-fold ({np.mean(manual):.4f}) vs. sklearn cross_val_score "
        f"({sklearn_equiv.mean():.4f}) on the same\npre-shuffled data: "
        f"{'match' if np.allclose(manual, sklearn_equiv) else 'MISMATCH'} (fold-by-fold)."
    )

    plt.figure(figsize=(7, 4))
    bins = np.linspace(min(single_split_scores.min(), cv_scores.min()) - 0.005, 1.0, 40)
    plt.hist(single_split_scores, bins=bins, alpha=0.65, label="Single 80/20 split")
    plt.hist(cv_scores, bins=bins, alpha=0.65, label="5-fold CV mean")
    plt.xlabel("Estimated test accuracy")
    plt.ylabel("Count over 200 random seeds")
    plt.title("Same data, same model — only the split changed")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "split_variance.png", dpi=120)
    plt.close()
    print("Saved plot to outputs/split_variance.png")


# ---------------------------------------------------------------------------
# Part 3 — bias-variance, decomposed numerically
# ---------------------------------------------------------------------------


TRUE_SIGMA = 0.35  # the irreducible noise we inject; nothing can beat this


def true_function(x: np.ndarray) -> np.ndarray:
    return np.sin(1.5 * x)


def sample_dataset(n: int, rng: np.random.Generator):
    x = rng.uniform(0, 5, size=n)
    y = true_function(x) + rng.normal(0, TRUE_SIGMA, size=n)
    return x, y


def run_bias_variance_demo() -> None:
    print()
    print("=" * 74)
    print("PART 3 — Bias-variance: verifying the decomposition adds up")
    print("=" * 74)

    # Degrees stop at 10: past that a polynomial on 30 points is so unstable that the
    # variance column runs to five figures, which crushes the plot's scale without
    # teaching anything the degree-9/10 rows don't already show. Exercise 3 lifts the cap.
    degrees = range(1, 11)
    n_datasets, n_train = 200, 30
    rng = np.random.default_rng(7)

    x_test = np.linspace(0.2, 4.8, 100)
    f_true = true_function(x_test)

    # Draw many training sets from the same underlying process. "Variance" is
    # literally the spread of the fitted predictions ACROSS these datasets — a
    # quantity you can only see by refitting, which is why it feels abstract when
    # you only ever fit once.
    datasets = [sample_dataset(n_train, rng) for _ in range(n_datasets)]

    rows = []
    for degree in degrees:
        preds = np.empty((n_datasets, len(x_test)))
        for b, (x_tr, y_tr) in enumerate(datasets):
            poly = np.polynomial.Polynomial.fit(x_tr, y_tr, deg=degree)
            preds[b] = poly(x_test)

        mean_pred = preds.mean(axis=0)  # E_D[f_D(x)]
        bias_sq = np.mean((mean_pred - f_true) ** 2)  # (10) systematic error
        variance = np.mean(preds.var(axis=0))  # (11) sensitivity to the sample
        noise = TRUE_SIGMA**2  # (12) irreducible

        # Measured error against FRESH noisy targets, averaged over datasets. This is
        # the left-hand side of the decomposition — computed independently of (10)-(12).
        y_test_noisy = f_true + rng.normal(0, TRUE_SIGMA, size=(n_datasets, len(x_test)))
        measured = np.mean((y_test_noisy - preds) ** 2)

        train_err = np.mean([
            np.mean((y_tr - np.polynomial.Polynomial.fit(x_tr, y_tr, deg=degree)(x_tr)) ** 2)
            for x_tr, y_tr in datasets
        ])
        rows.append((degree, bias_sq, variance, noise, bias_sq + variance + noise, measured, train_err))

    print(f"\n{'deg':>4}{'bias²':>10}{'variance':>11}{'noise':>9}{'predicted':>12}{'measured':>11}{'train err':>11}")
    print("-" * 74)
    for degree, b2, v, nz, predicted, measured, tr in rows:
        print(f"{degree:>4}{b2:>10.4f}{v:>11.4f}{nz:>9.4f}{predicted:>12.4f}{measured:>11.4f}{tr:>11.4f}")

    worst_gap = max(abs(p - m) for _, _, _, _, p, m, _ in rows)
    best_degree = min(rows, key=lambda r: r[5])[0]
    print(
        f"\n'predicted' (bias² + variance + noise) tracks 'measured' to within {worst_gap:.4f} at every\n"
        f"degree — the decomposition is an identity, not a metaphor. Reading the columns:\n"
        f"  - bias² falls as the model gets more flexible (degree 1 can't bend into a sine).\n"
        f"  - variance rises as it gets more flexible (degree 15 chases the noise in its\n"
        f"    particular 30-point sample, so it swings wildly from dataset to dataset).\n"
        f"  - test error is their sum plus {TRUE_SIGMA**2:.4f} of noise, so it's U-shaped, minimised at\n"
        f"    degree {best_degree}. Training error, meanwhile, just keeps falling — which is exactly\n"
        f"    why you cannot pick a model by looking at training error."
    )

    degs = [r[0] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(degs, [r[1] for r in rows], marker="o", label="bias²")
    ax1.plot(degs, [r[2] for r in rows], marker="s", label="variance")
    ax1.plot(degs, [r[4] for r in rows], marker="^", label="bias² + var + noise")
    ax1.axhline(TRUE_SIGMA**2, color="grey", linestyle=":", label="irreducible noise")
    ax1.axvline(best_degree, color="black", linewidth=1, linestyle="--")
    ax1.set_xlabel("Polynomial degree (model flexibility)")
    ax1.set_ylabel("Squared error")
    ax1.set_title("The tradeoff, decomposed")
    ax1.set_yscale("log")
    ax1.legend(fontsize=8)

    ax2.plot(degs, [r[6] for r in rows], marker="o", label="Training error")
    ax2.plot(degs, [r[5] for r in rows], marker="s", label="Test error (measured)")
    ax2.axhline(TRUE_SIGMA**2, color="grey", linestyle=":", label="irreducible noise floor")
    ax2.axvline(best_degree, color="black", linewidth=1, linestyle="--", label=f"best degree ({best_degree})")
    ax2.set_xlabel("Polynomial degree (model flexibility)")
    ax2.set_ylabel("Mean squared error")
    ax2.set_title("Underfitting → sweet spot → overfitting")
    ax2.set_yscale("log")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bias_variance.png", dpi=120)
    plt.close()
    print("Saved plot to outputs/bias_variance.png")


# ---------------------------------------------------------------------------
# Part 4 — ROC vs. PR, and what AUC actually means
# ---------------------------------------------------------------------------


def run_roc_pr_demo() -> None:
    print()
    print("=" * 74)
    print("PART 4 — ROC-AUC vs. PR-AUC, and a Monte-Carlo proof of what AUC means")
    print("=" * 74)

    X, y = make_imbalanced()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]

    fpr, tpr = roc_curve_scratch(y_test, scores)
    recall, precision = pr_curve_scratch(y_test, scores)
    roc_auc = auc_trapezoid(fpr, tpr)
    pr_auc = average_precision_scratch(y_test, scores)
    pr_auc_trap = auc_trapezoid(recall, precision)
    prevalence = y_test.mean()

    print(f"\nROC-AUC (scratch): {roc_auc:.4f}   sklearn roc_auc_score:      "
          f"{roc_auc_score(y_test, scores):.4f}")
    print(f"PR-AUC  (scratch): {pr_auc:.4f}   sklearn average_precision:  "
          f"{average_precision_score(y_test, scores):.4f}")
    print(
        f"\nNote the PR curve has two different 'areas': the step-wise average precision above\n"
        f"({pr_auc:.4f}) and the trapezoid rule applied to the same points ({pr_auc_trap:.4f}). They\n"
        f"disagree because trapezoid interpolation on a PR curve invents operating points no\n"
        f"threshold can actually produce. Average precision is the one to report."
    )
    print(
        f"\nA random-guessing baseline scores 0.5 on ROC but only {prevalence:.3f} (the prevalence)\n"
        f"on PR. So this model's ROC-AUC of {roc_auc:.2f} sits {roc_auc - 0.5:.2f} above its baseline, while its\n"
        f"PR-AUC of {pr_auc:.2f} sits {pr_auc - prevalence:.2f} above its own. ROC's x-axis is FPR = FP/(FP+TN),\n"
        f"and TN is huge when negatives dominate — so hundreds of false positives barely move\n"
        f"the curve. PR has no TN term anywhere, which is why it stays honest under imbalance."
    )

    # What does AUC *mean*? Claim: ROC-AUC = P(score of a random positive > score of a
    # random negative), ties counting half. Test it by sampling pairs directly.
    pos_scores = scores[y_test == 1]
    neg_scores = scores[y_test == 0]
    n_pairs = 200_000
    i = RNG.integers(0, len(pos_scores), n_pairs)
    j = RNG.integers(0, len(neg_scores), n_pairs)
    wins = np.sum(pos_scores[i] > neg_scores[j]) + 0.5 * np.sum(pos_scores[i] == neg_scores[j])
    mc_estimate = wins / n_pairs
    print(
        f"\nMonte-Carlo check on {n_pairs:,} random (positive, negative) pairs:\n"
        f"  P(positive ranked above negative) = {mc_estimate:.4f}\n"
        f"  ROC-AUC from the curve            = {roc_auc:.4f}    difference: {abs(mc_estimate - roc_auc):.4f}\n"
        f"They agree because they are the same quantity. That's the useful way to read AUC:\n"
        f"it's a *ranking* score, independent of any threshold — which is also its limitation,\n"
        f"since a model can rank perfectly and still be badly calibrated as a probability."
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(fpr, tpr, label=f"Model (AUC = {roc_auc:.3f})")
    ax1.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random guessing (0.500)")
    ax1.set_xlabel("False positive rate — FP/(FP+TN)")
    ax1.set_ylabel("True positive rate — recall")
    ax1.set_title("ROC: comfortably above the diagonal")
    ax1.legend(fontsize=8)

    ax2.plot(recall, precision, color="darkorange", label=f"Model (AP = {pr_auc:.3f})")
    ax2.axhline(prevalence, linestyle="--", color="grey", label=f"Random guessing ({prevalence:.3f})")
    ax2.set_xlabel("Recall — TP/(TP+FN)")
    ax2.set_ylabel("Precision — TP/(TP+FP)")
    ax2.set_ylim(0, 1.02)
    ax2.set_title("Precision-Recall: the honest picture at 3% prevalence")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_vs_pr.png", dpi=120)
    plt.close()
    print("Saved plot to outputs/roc_vs_pr.png")


# ---------------------------------------------------------------------------
# Part 5 — leakage: 75% accuracy out of pure noise
# ---------------------------------------------------------------------------


def run_leakage_demo() -> None:
    print()
    print("=" * 74)
    print("PART 5 — Data leakage: high accuracy from data with no signal at all")
    print("=" * 74)

    # There is NO relationship here. X is noise, y is a coin flip. The only honest
    # answer any evaluation can give is ~50%.
    n_samples, n_features, n_selected = 100, 5000, 20
    X = RNG.normal(size=(n_samples, n_features))
    y = RNG.integers(0, 2, size=n_samples)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))

    def top_features(Xs, ys, k):
        """Pick the k features most correlated with the labels."""
        Xc = Xs - Xs.mean(axis=0)
        yc = ys - ys.mean()
        denom = np.sqrt((Xc**2).sum(axis=0) * (yc**2).sum()) + 1e-12
        corr = np.abs((Xc * yc[:, None]).sum(axis=0) / denom)
        return np.argsort(-corr)[:k]

    # Repeat the whole experiment on fresh noise each time. A single 100-sample CV
    # estimate is itself noisy (Part 2's lesson applied to Part 5), so one run can
    # land at 41% or 58% by luck; averaging over datasets shows where each procedure
    # actually sits.
    n_repeats = 25
    leaky_runs, honest_runs = [], []
    for _ in range(n_repeats):
        X = RNG.normal(size=(n_samples, n_features))
        y = RNG.integers(0, 2, size=n_samples)
        cv = StratifiedKFold(5, shuffle=True, random_state=0)

        # WRONG: select features using the whole dataset, then cross-validate. The
        # selection step already looked at every label, including the ones each fold
        # is about to be tested on — so the "held-out" folds were never held out.
        leaky_idx = top_features(X, y, n_selected)
        leaky_runs.append(cross_val_score(model, X[:, leaky_idx], y, cv=cv).mean())

        # RIGHT: the selection is part of the model, so it belongs inside the fold.
        fold_scores = []
        for train_idx, val_idx in cv.split(X, y):
            idx = top_features(X[train_idx], y[train_idx], n_selected)  # training labels only
            model.fit(X[train_idx][:, idx], y[train_idx])
            fold_scores.append(accuracy_scratch(y[val_idx], model.predict(X[val_idx][:, idx])))
        honest_runs.append(float(np.mean(fold_scores)))

    leaky_score = float(np.mean(leaky_runs))
    honest_score = float(np.mean(honest_runs))

    print(f"\n{n_samples} samples, {n_features} pure-noise features, random coin-flip labels.")
    print(f"Ground truth: no model can beat 50%. Averaged over {n_repeats} fresh datasets.\n")
    print(f"{'Procedure':<52}{'CV accuracy':>12}")
    print("-" * 74)
    print(f"{'Select features on ALL data, then cross-validate':<52}{leaky_score:>12.3f}   <- leaked")
    print(f"{'Select features inside each fold (train only)':<52}{honest_score:>12.3f}   <- honest")
    print(
        f"\nThe leaky procedure reports {leaky_score:.0%} on data containing no signal whatsoever.\n"
        f"With {n_features} random features and {n_samples} samples, some features correlate with the labels\n"
        f"by chance alone; choosing them using all the labels bakes the test set's answers into\n"
        f"the feature set. The honest version lands at {honest_score:.0%} — chance, which is the truth.\n\n"
        f"The rule this gives you: cross-validation only protects you if EVERY step that\n"
        f"touches labels — selection, scaling, imputation, resampling, threshold tuning —\n"
        f"happens inside the fold. That's the real reason to use a Pipeline: it makes the\n"
        f"preprocessing part of the model, so cross_val_score refits it per fold automatically."
    )

    plt.figure(figsize=(6.5, 4))
    bars = plt.bar(
        ["Leaky\n(select on all data)", "Honest\n(select inside fold)"],
        [leaky_score, honest_score],
        color=["indianred", "seagreen"],
    )
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Truth: no signal (50%)")
    for bar, val in zip(bars, [leaky_score, honest_score]):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.1%}", ha="center")
    plt.ylabel("Reported cross-validated accuracy")
    plt.ylim(0, 1.0)
    plt.title("Same data, same model, same CV — only the order of operations changed")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "data_leakage.png", dpi=120)
    plt.close()
    print("Saved plot to outputs/data_leakage.png")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    run_accuracy_paradox_demo()
    run_split_variance_demo()
    run_bias_variance_demo()
    run_roc_pr_demo()
    run_leakage_demo()
