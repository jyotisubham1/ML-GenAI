"""
Clustering & Dimensionality Reduction — the first unsupervised project. k-means and
PCA implemented from scratch from their objective functions, with six experiments:

  Part 1  k-means from scratch; the objective provably never increases
  Part 2  local minima are real — random init vs k-means++ over 200 restarts
  Part 3  choosing k without labels: the elbow and the silhouette
  Part 4  where k-means structurally fails: non-convex clusters
  Part 5  PCA from scratch via eigendecomposition, checked against sklearn
  Part 6  PCA is provably the best linear projection — tested against 500 random ones
  Part 7  putting it together: compressing handwritten digits

Run:
    python clustering_and_pca.py

See README.md for the math behind every formula referenced in the comments below.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write plots to file; don't require a display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.datasets import load_digits, make_blobs, make_moons
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

RNG = np.random.default_rng(seed=42)
OUTPUT_DIR = Path(__file__).parent / "outputs"


# ---------------------------------------------------------------------------
# k-means, from scratch
# ---------------------------------------------------------------------------


class KMeansScratch:
    """
    Minimizes the within-cluster sum of squares (WCSS, a.k.a. inertia):

        J = sum_i || x_i - mu_{c_i} ||^2                                       (1)

    by alternating two steps, each of which minimizes J while holding the other
    variable fixed (Lloyd's algorithm):

        assign:  c_i  <- argmin_k || x_i - mu_k ||^2                           (2)
        update:  mu_k <- mean of the points currently assigned to k            (3)

    Neither step can ever increase J (README §4.2), so J decreases monotonically
    and the algorithm must converge. It converges to a LOCAL minimum, though —
    which is what Part 2 is about.
    """

    def __init__(self, n_clusters: int = 3, max_iter: int = 300, tol: float = 1e-9,
                 init: str = "k-means++", seed: int | None = None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.rng = np.random.default_rng(seed)
        self.centroids: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float = np.inf
        self.objective_history_: list[float] = []

    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        if self.init == "random":
            # Pick k data points uniformly at random. Simple, and often bad — Part 2.
            idx = self.rng.choice(len(X), self.n_clusters, replace=False)
            return X[idx].copy()

        # k-means++: pick the first centre at random, then pick each subsequent centre
        # with probability proportional to its squared distance from the nearest centre
        # already chosen. Far-away points are far more likely to be picked, so the
        # initial centres start spread out instead of clumped. See README §4.3.
        centroids = [X[self.rng.integers(len(X))]]
        for _ in range(1, self.n_clusters):
            d2 = np.min(
                ((X[:, None, :] - np.array(centroids)[None, :, :]) ** 2).sum(axis=2), axis=1
            )
            total = d2.sum()
            probs = d2 / total if total > 0 else np.full(len(X), 1 / len(X))
            centroids.append(X[self.rng.choice(len(X), p=probs)])
        return np.array(centroids)

    @staticmethod
    def _distances(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Squared euclidean distance from every point to every centroid: (n, k)."""
        return ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)

    def fit(self, X: np.ndarray) -> "KMeansScratch":
        self.centroids = self._init_centroids(X)
        self.objective_history_ = []

        for _ in range(self.max_iter):
            d2 = self._distances(X, self.centroids)
            labels = np.argmin(d2, axis=1)  # (2) assignment step
            # J is the distance to the centroid each point was actually assigned to.
            objective = float(d2[np.arange(len(X)), labels].sum())  # (1)
            self.objective_history_.append(objective)

            new_centroids = self.centroids.copy()
            for k in range(self.n_clusters):
                members = X[labels == k]
                if len(members) > 0:
                    new_centroids[k] = members.mean(axis=0)  # (3) update step
                # An empty cluster keeps its old centroid — rare, and re-seeding it
                # is an implementation choice sklearn makes differently.

            shift = np.abs(new_centroids - self.centroids).max()
            self.centroids = new_centroids
            if shift < self.tol:
                break

        d2 = self._distances(X, self.centroids)
        self.labels_ = np.argmin(d2, axis=1)
        self.inertia_ = float(d2[np.arange(len(X)), self.labels_].sum())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmin(self._distances(X, self.centroids), axis=1)


# ---------------------------------------------------------------------------
# PCA, from scratch
# ---------------------------------------------------------------------------


class PCAScratch:
    """
    Principal Component Analysis via eigendecomposition of the covariance matrix.

        centre:      X_c = X - mean(X)                                         (4)
        covariance:  C   = (1/n) X_c^T X_c                                     (5)
        solve:       C w = lambda w                                            (6)

    (6) says the directions we want are the EIGENVECTORS of C, and each one's
    eigenvalue lambda IS the variance captured along it. README §5.2 derives this
    from "maximize variance subject to ||w|| = 1" using a Lagrange multiplier —
    the eigenvector equation is not assumed, it falls out.

    Note (5) divides by n, not n-1. That makes the identity in Part 6 exact:
    reconstruction error using k components equals the sum of the DISCARDED
    eigenvalues. sklearn reports explained_variance_ with n-1; the ratios agree.
    """

    def __init__(self, n_components: int | None = None):
        self.n_components = n_components
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "PCAScratch":
        self.mean_ = X.mean(axis=0)  # (4)
        X_c = X - self.mean_
        n = len(X)
        C = (X_c.T @ X_c) / n  # (5)

        # eigh (not eig) because C is symmetric: guarantees real eigenvalues and
        # orthogonal eigenvectors, and is faster and more numerically stable.
        eigenvalues, eigenvectors = np.linalg.eigh(C)  # (6)

        order = np.argsort(eigenvalues)[::-1]  # eigh returns ascending; we want descending
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

        k = self.n_components or X.shape[1]
        self.explained_variance_ = eigenvalues[:k]
        self.components_ = eigenvectors[:, :k].T  # rows = components
        self.all_eigenvalues_ = eigenvalues
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project onto the components: the new coordinates of each point."""
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Map back to the original space — lossy unless every component is kept."""
        return Z @ self.components_ + self.mean_

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        return self.explained_variance_ / self.all_eigenvalues_.sum()


# ---------------------------------------------------------------------------
# Part 1 — k-means from scratch
# ---------------------------------------------------------------------------


def run_kmeans_demo() -> None:
    print("=" * 74)
    print("PART 1 — k-means from scratch: the objective can only go down")
    print("=" * 74)

    X, y_true = make_blobs(n_samples=600, centers=4, cluster_std=1.1, random_state=42)

    model = KMeansScratch(n_clusters=4, init="k-means++", seed=0).fit(X)
    history = model.objective_history_

    print(f"\nConverged in {len(history)} iterations.")
    print(f"{'iteration':>10}{'objective J':>16}{'change':>14}")
    print("-" * 74)
    for i, obj in enumerate(history):
        change = "" if i == 0 else f"{obj - history[i-1]:+.4f}"
        if i < 6 or i == len(history) - 1:
            print(f"{i:>10}{obj:>16.4f}{change:>14}")

    increases = [i for i in range(1, len(history)) if history[i] > history[i - 1] + 1e-9]
    print(
        f"\nJ never increased: {len(increases)} of {len(history)-1} steps went up. That is not "
        f"luck —\nboth steps of the algorithm minimize the SAME objective J (README §4.2), so "
        f"each\none can only lower it or leave it alone. Since there are finitely many ways to\n"
        f"partition the points, a strictly decreasing sequence must terminate: k-means is\n"
        f"guaranteed to converge."
    )

    sk = SklearnKMeans(n_clusters=4, n_init=10, random_state=42).fit(X)
    print(f"\n{'Implementation':<34}{'inertia (J)':>14}{'agreement (ARI)':>18}")
    print("-" * 74)
    print(f"{'Scratch k-means':<34}{model.inertia_:>14.3f}"
          f"{adjusted_rand_score(y_true, model.labels_):>18.4f}")
    print(f"{'sklearn KMeans':<34}{sk.inertia_:>14.3f}"
          f"{adjusted_rand_score(y_true, sk.labels_):>18.4f}")
    print(
        "\nARI (adjusted Rand index) compares two clusterings while ignoring which label\n"
        "number got attached to which group — 1.0 is a perfect match, 0.0 is chance. We\n"
        "only have the true labels here because the data is synthetic; in a real\n"
        "clustering problem nobody hands you the answer, which is the whole difficulty."
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(history, marker="o", markersize=3)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Objective J (within-cluster sum of squares)")
    ax1.set_title("Monotonically decreasing — provably")
    ax2.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap="viridis", s=12, alpha=0.7)
    ax2.scatter(model.centroids[:, 0], model.centroids[:, 1], c="red", marker="X",
                s=200, edgecolor="black", linewidth=1, label="centroids")
    ax2.set_title("Final clusters and centroids")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "kmeans_convergence.png", dpi=120)
    plt.close()
    print("Saved plot to outputs/kmeans_convergence.png")


# ---------------------------------------------------------------------------
# Part 2 — local minima and k-means++
# ---------------------------------------------------------------------------


def run_initialization_demo() -> None:
    print()
    print("=" * 74)
    print("PART 2 — Local minima are real: random init vs. k-means++")
    print("=" * 74)

    X, _ = make_blobs(n_samples=600, centers=6, cluster_std=1.0, random_state=7)
    n_restarts = 200

    results = {}
    for init in ("random", "k-means++"):
        inertias = [
            KMeansScratch(n_clusters=6, init=init, seed=s).fit(X).inertia_
            for s in range(n_restarts)
        ]
        results[init] = np.array(inertias)

    best = min(r.min() for r in results.values())
    print(f"\n{n_restarts} restarts each, same data, k=6. Best J found overall: {best:.2f}\n")
    print(f"{'Init':<16}{'best J':>10}{'median J':>11}{'worst J':>10}{'% stuck >1% above best':>25}")
    print("-" * 74)
    for init, arr in results.items():
        stuck = 100 * np.mean(arr > best * 1.01)
        print(f"{init:<16}{arr.min():>10.2f}{np.median(arr):>11.2f}{arr.max():>10.2f}{stuck:>24.1f}%")

    print(
        f"\nSame algorithm, same data — only the starting centroids differ. Random init got\n"
        f"stuck in a clearly worse solution {100 * np.mean(results['random'] > best * 1.01):.0f}% of the time; "
        f"k-means++ "
        f"{100 * np.mean(results['k-means++'] > best * 1.01):.0f}% of the time.\n"
        f"This is what 'converges to a LOCAL minimum' costs you in practice, and it is why\n"
        f"sklearn runs k-means 10 times by default (n_init=10) and keeps the best J. Note\n"
        f"you can only do that because J is computable WITHOUT labels — you can always\n"
        f"tell which of two clusterings scored better, even with no ground truth."
    )

    plt.figure(figsize=(7, 4.2))
    bins = np.linspace(min(r.min() for r in results.values()) * 0.99,
                       max(r.max() for r in results.values()) * 1.01, 45)
    for init, arr in results.items():
        plt.hist(arr, bins=bins, alpha=0.6, label=f"{init} (median {np.median(arr):.0f})")
    plt.axvline(best, color="black", linestyle="--", linewidth=1, label=f"best found ({best:.0f})")
    plt.xlabel("Final objective J after convergence")
    plt.ylabel(f"Count over {n_restarts} restarts")
    plt.title("Where k-means lands depends on where it starts")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "kmeans_initialization.png", dpi=120)
    plt.close()
    print("Saved plot to outputs/kmeans_initialization.png")


# ---------------------------------------------------------------------------
# Part 3 — choosing k
# ---------------------------------------------------------------------------


def run_choosing_k_demo() -> None:
    print()
    print("=" * 74)
    print("PART 3 — Choosing k when nobody tells you the answer")
    print("=" * 74)

    true_k = 5
    X, _ = make_blobs(n_samples=700, centers=true_k, cluster_std=1.0, random_state=3)

    ks = list(range(2, 11))
    inertias, silhouettes = [], []
    for k in ks:
        # Best of 10 restarts, exactly as Part 2 concluded you must. With a single
        # start, an unlucky local minimum at one k makes the whole curve unreadable —
        # inertia can even come out LOWER at k+1 than at k, which is impossible for
        # properly minimized J. Try it by setting n_init = 1.
        n_init = 10
        candidates = [
            KMeansScratch(n_clusters=k, init="k-means++", seed=s).fit(X) for s in range(n_init)
        ]
        model = min(candidates, key=lambda m: m.inertia_)
        inertias.append(model.inertia_)
        silhouettes.append(silhouette_score(X, model.labels_))

    print(f"\nData actually has {true_k} clusters (we generated it, so we know).\n")
    print(f"{'k':>4}{'inertia J':>14}{'drop from k-1':>16}{'silhouette':>13}")
    print("-" * 74)
    for i, k in enumerate(ks):
        drop = "" if i == 0 else f"{inertias[i-1] - inertias[i]:.1f}"
        print(f"{k:>4}{inertias[i]:>14.1f}{drop:>16}{silhouettes[i]:>13.4f}")

    best_sil = ks[int(np.argmax(silhouettes))]
    print(
        f"\nInertia ALWAYS falls as k rises — at k = n every point is its own cluster and\n"
        f"J = 0, so you cannot simply minimize it. The 'elbow' is where the drops stop\n"
        f"being dramatic: look at the 'drop from k-1' column and find where it collapses.\n"
        f"\nSilhouette needs no such eyeballing. For each point it compares a = mean distance\n"
        f"to its OWN cluster against b = mean distance to the nearest OTHER cluster, scoring\n"
        f"(b - a)/max(a, b): +1 means comfortably inside its cluster, 0 means on a border,\n"
        f"negative means it is probably in the wrong one. It peaks at k = {best_sil}"
        f"{' — the truth' if best_sil == true_k else f' (truth is {true_k})'}."
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(ks, inertias, marker="o")
    ax1.axvline(true_k, color="black", linestyle="--", linewidth=1, label=f"true k = {true_k}")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia J")
    ax1.set_title("Elbow method — find where the drop flattens")
    ax1.legend(fontsize=8)
    ax2.plot(ks, silhouettes, marker="s", color="darkorange")
    ax2.axvline(true_k, color="black", linestyle="--", linewidth=1, label=f"true k = {true_k}")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Mean silhouette score")
    ax2.set_title("Silhouette — higher is better, no eyeballing needed")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "choosing_k.png", dpi=120)
    plt.close()
    print("Saved plot to outputs/choosing_k.png")


# ---------------------------------------------------------------------------
# Part 4 — where k-means fails
# ---------------------------------------------------------------------------


def run_failure_demo() -> None:
    print()
    print("=" * 74)
    print("PART 4 — Where k-means structurally fails")
    print("=" * 74)

    moons, moon_labels = make_moons(n_samples=600, noise=0.06, random_state=42)

    # Two blobs with very different spreads: k-means splits the wide one and
    # merges part of it with the narrow one, because it only compares distances.
    tight = RNG.normal(loc=[0, 0], scale=0.35, size=(300, 2))
    wide = RNG.normal(loc=[4, 0], scale=2.0, size=(300, 2))
    uneven = np.vstack([tight, wide])
    uneven_labels = np.r_[np.zeros(300), np.ones(300)]

    cases = [("Two moons (non-convex)", moons, moon_labels),
             ("Unequal spread", uneven, uneven_labels)]

    print(f"\n{'Dataset':<28}{'ARI vs. true grouping':>24}")
    print("-" * 74)
    fitted = []
    for name, data, labels in cases:
        model = KMeansScratch(n_clusters=2, init="k-means++", seed=0).fit(data)
        fitted.append((name, data, model))
        print(f"{name:<28}{adjusted_rand_score(labels, model.labels_):>24.4f}")

    print(
        "\nBoth failures come from the same source, and it is in the objective, not the\n"
        "algorithm. J only ever measures SQUARED DISTANCE TO A CENTRE, so k-means can only\n"
        "carve space into regions nearest to each centre — which are always convex blobs\n"
        "of roughly equal spread (formally: a Voronoi partition).\n"
        "  - Two moons: each crescent's own centre lies OUTSIDE it, nearer the other\n"
        "    crescent. No placement of two centres can recover the shapes.\n"
        "  - Unequal spread: the wide cluster's outer points are closer to the tight\n"
        "    cluster's centre than to their own, so they get reassigned.\n"
        "This is not a bug to fix by tuning. It means the assumption baked into J is wrong\n"
        "for this data, so you need a different objective — density-based (DBSCAN),\n"
        "distribution-based (Gaussian mixtures), or graph-based (spectral clustering)."
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for row, (name, data, model) in enumerate(fitted):
        axes[row, 0].scatter(data[:, 0], data[:, 1], c=cases[row][2], cmap="coolwarm", s=10)
        axes[row, 0].set_title(f"{name}: the grouping we want", fontsize=10)
        axes[row, 1].scatter(data[:, 0], data[:, 1], c=model.labels_, cmap="coolwarm", s=10)
        axes[row, 1].scatter(model.centroids[:, 0], model.centroids[:, 1], c="black",
                             marker="X", s=150)
        axes[row, 1].set_title(f"{name}: what k-means finds", fontsize=10)
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "kmeans_failures.png", dpi=120)
    plt.close()
    print("Saved plot to outputs/kmeans_failures.png")


# ---------------------------------------------------------------------------
# Parts 5 & 6 — PCA
# ---------------------------------------------------------------------------


def run_pca_demo() -> None:
    print()
    print("=" * 74)
    print("PART 5 & 6 — PCA from scratch, and why it is the optimal linear projection")
    print("=" * 74)

    digits = load_digits()
    X = digits.data  # (1797, 64) — 8x8 greyscale handwritten digits

    scratch = PCAScratch().fit(X)
    sk = SklearnPCA().fit(X)

    print(f"\nData: {X.shape[0]} handwritten digits, each an 8x8 image = {X.shape[1]} features.")
    print(f"\n{'Component':<12}{'scratch var ratio':>20}{'sklearn var ratio':>20}{'match':>10}")
    print("-" * 74)
    for i in range(5):
        a, b = scratch.explained_variance_ratio_[i], sk.explained_variance_ratio_[i]
        print(f"{'PC' + str(i+1):<12}{a:>20.6f}{b:>20.6f}{'ok' if np.isclose(a, b) else 'MISMATCH':>10}")

    cum = np.cumsum(scratch.explained_variance_ratio_)
    n_for = {p: int(np.searchsorted(cum, p) + 1) for p in (0.5, 0.8, 0.9, 0.95, 0.99)}
    print(f"\nComponents needed to retain a given share of the total variance:")
    for p, n in n_for.items():
        print(f"  {p:>4.0%} of variance -> {n:>2} of 64 components  "
              f"({100 * n / 64:.0f}% of the original size)")
    print(
        f"\n{n_for[0.9]} numbers reproduce 90% of the variation in a 64-pixel image. The pixels\n"
        f"are enormously redundant — neighbouring pixels are nearly always similar, and\n"
        f"corners are almost always blank — and PCA finds and removes exactly that\n"
        f"redundancy."
    )

    # --- The identity: reconstruction error == sum of the discarded eigenvalues ---
    print(f"\n{'k':>4}{'reconstruction MSE':>22}{'sum of dropped eigenvalues':>30}")
    print("-" * 74)
    for k in (2, 8, 20, 40):
        p = PCAScratch(n_components=k).fit(X)
        recon = p.inverse_transform(p.transform(X))
        measured = float(np.mean(((X - recon) ** 2).sum(axis=1)))
        predicted = float(p.all_eigenvalues_[k:].sum())
        print(f"{k:>4}{measured:>22.6f}{predicted:>30.6f}")
    print(
        "\nThese two columns are computed completely differently — one reconstructs every\n"
        "image and measures the error, the other just adds up eigenvalues — and they agree\n"
        "to six decimals. This is the theorem of §5.3, and the agreement confirms the\n"
        "implementation. (It is an identity, so this checks the CODE, not the maths.)"
    )

    # --- A genuine empirical test: is PCA really the BEST k-dim linear projection? ---
    k = 10
    p = PCAScratch(n_components=k).fit(X)
    pca_err = float(np.mean(((X - p.inverse_transform(p.transform(X))) ** 2).sum(axis=1)))

    X_c = X - X.mean(axis=0)
    n_random, wins = 500, 0
    random_errs = []
    for _ in range(n_random):
        # A random orthonormal k-dimensional basis, via QR of a random matrix.
        Q, _ = np.linalg.qr(RNG.normal(size=(X.shape[1], k)))
        recon = (X_c @ Q) @ Q.T
        err = float(np.mean(((X_c - recon) ** 2).sum(axis=1)))
        random_errs.append(err)
        if err < pca_err:
            wins += 1

    print(f"\nIs PCA really optimal? Testing k={k} against {n_random} random orthonormal bases:")
    print(f"  PCA reconstruction error            : {pca_err:.4f}")
    print(f"  Best of {n_random} random projections     : {min(random_errs):.4f}")
    print(f"  Median random projection            : {np.median(random_errs):.4f}")
    print(f"  Random projections that beat PCA    : {wins} / {n_random}")
    print(
        f"\nNot one of {n_random} random projections beat it, and the best was "
        f"{min(random_errs)/pca_err:.1f}x worse.\nThat is the Eckart-Young theorem made "
        f"empirical: among ALL k-dimensional linear\nprojections, the one spanned by the top-k "
        f"eigenvectors minimizes reconstruction\nerror. Unlike the identity above, this "
        f"comparison could have come out otherwise."
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.bar(range(1, 21), scratch.explained_variance_ratio_[:20])
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Share of total variance")
    ax1.set_title("Scree plot — each component's contribution")
    ax2.plot(range(1, 65), cum, marker="o", markersize=3)
    for p_, style in [(0.9, "--"), (0.95, ":")]:
        ax2.axhline(p_, color="grey", linestyle=style, linewidth=1)
        ax2.axvline(n_for[p_], color="grey", linestyle=style, linewidth=1)
    ax2.set_xlabel("Number of components kept")
    ax2.set_ylabel("Cumulative variance retained")
    ax2.set_title(f"{n_for[0.9]} of 64 components hold 90% of the variance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pca_variance.png", dpi=120)
    plt.close()
    print("Saved plot to outputs/pca_variance.png")


# ---------------------------------------------------------------------------
# Part 7 — compression, visually
# ---------------------------------------------------------------------------


def run_compression_demo() -> None:
    print()
    print("=" * 74)
    print("PART 7 — What discarding components actually looks like")
    print("=" * 74)

    digits = load_digits()
    X, y = digits.data, digits.target
    ks = [1, 2, 5, 10, 20, 40, 64]

    fig, axes = plt.subplots(len(ks) + 1, 8, figsize=(9, 1.15 * (len(ks) + 1)))
    for col in range(8):
        axes[0, col].imshow(X[col].reshape(8, 8), cmap="gray_r")
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])
    axes[0, 0].set_ylabel("original", fontsize=7, rotation=0, ha="right", va="center")

    print(f"\n{'k':>4}{'reconstruction MSE':>22}{'variance kept':>16}{'compression':>14}")
    print("-" * 74)
    for row, k in enumerate(ks, start=1):
        p = PCAScratch(n_components=k).fit(X)
        recon = p.inverse_transform(p.transform(X))
        mse = float(np.mean(((X - recon) ** 2).sum(axis=1)))
        kept = float(p.explained_variance_ratio_.sum())
        print(f"{k:>4}{mse:>22.3f}{kept:>15.1%}{f'{64/k:.1f}x':>14}")
        for col in range(8):
            axes[row, col].imshow(recon[col].reshape(8, 8), cmap="gray_r")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        axes[row, 0].set_ylabel(f"k={k}", fontsize=7, rotation=0, ha="right", va="center")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pca_reconstructions.png", dpi=130)
    plt.close()

    # Does clustering survive compression? Unsupervised end-to-end.
    print(f"\n{'Clustering input':<38}{'ARI vs. true digit':>20}{'dims':>10}")
    print("-" * 74)
    full = SklearnKMeans(n_clusters=10, n_init=10, random_state=0).fit(X)
    print(f"{'k-means on all 64 pixels':<38}{adjusted_rand_score(y, full.labels_):>20.4f}{64:>10}")
    for k in (10, 20):
        Z = PCAScratch(n_components=k).fit(X).transform(X)
        km = SklearnKMeans(n_clusters=10, n_init=10, random_state=0).fit(Z)
        print(f"{f'k-means on {k} principal components':<38}"
              f"{adjusted_rand_score(y, km.labels_):>20.4f}{k:>10}")

    print(
        "\nClustering quality survives — sometimes improves — after throwing away most of\n"
        "the dimensions, because what PCA discards is mostly noise and redundancy. This is\n"
        "the practical reason to reduce dimensions before clustering: distance-based\n"
        "methods degrade in high dimensions (the 'curse of dimensionality'), so removing\n"
        "uninformative axes can make the distances more meaningful, not less.\n"
        "Nothing here used the digit labels — they were only for scoring at the end."
    )
    print("Saved plot to outputs/pca_reconstructions.png")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    run_kmeans_demo()
    run_initialization_demo()
    run_choosing_k_demo()
    run_failure_demo()
    run_pca_demo()
    run_compression_demo()
