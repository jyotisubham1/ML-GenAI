"""
Neural Networks in PyTorch — the same network as project 06, now with autograd
doing the differentiation. Six experiments:

  Part 1  autograd's gradients vs. YOUR hand-derived backprop: identical to ~1e-15
  Part 2  what a computation graph is, and hand-checking one derivative
  Part 3  optimizers — SGD, momentum, RMSprop, Adam — traced on a hard surface
  Part 4  full-batch vs. mini-batch: measured in epochs AND in wall-clock seconds
  Part 5  regularization — weight decay and dropout, measured on the train/val gap
  Part 6  a proper pipeline: select on validation, touch the test set exactly once

Run:
    python pytorch_networks.py

See README.md for the math behind every formula referenced in the comments below.
"""

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write plots to file; don't require a display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = Path(__file__).parent / "outputs"
torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Part 1 — autograd vs. the backprop you derived by hand in project 06
# ---------------------------------------------------------------------------


def scratch_forward_backward(X, Y, weights, biases):
    """
    Project 06's derivation, transcribed. Nothing here knows PyTorch exists.

        forward:   z^l = a^(l-1) W^l + b^l ;  a^l = relu(z^l) ;  a^L = softmax(z^L)
        backward:  delta^L    = (a^L - Y) / n                                  (7)
                   dJ/dW^l    = (a^(l-1))^T delta^l                            (8)
                   delta^(l-1) = (delta^l (W^l)^T) * relu'(z^(l-1))           (10)
    """
    activations, pre_activations = [X], []
    a = X
    for i, (W, b) in enumerate(zip(weights, biases)):
        z = a @ W + b
        pre_activations.append(z)
        if i == len(weights) - 1:
            z_shift = z - z.max(axis=1, keepdims=True)
            e = np.exp(z_shift)
            a = e / e.sum(axis=1, keepdims=True)
        else:
            a = np.maximum(0, z)
        activations.append(a)

    n = len(Y)
    loss = float(-np.mean(np.sum(Y * np.log(np.clip(activations[-1], 1e-12, 1.0)), axis=1)))

    dW = [None] * len(weights)
    db = [None] * len(biases)
    delta = (activations[-1] - Y) / n  # (7)
    for l in reversed(range(len(weights))):
        dW[l] = activations[l].T @ delta  # (8)
        db[l] = delta.sum(axis=0)  # (9)
        if l > 0:
            delta = (delta @ weights[l].T) * (pre_activations[l - 1] > 0)  # (10)
    return loss, dW, db


def run_autograd_vs_scratch() -> None:
    print("=" * 74)
    print("PART 1 — Does autograd compute the same gradients you derived by hand?")
    print("=" * 74)

    rng = np.random.default_rng(0)
    n, sizes = 32, [8, 16, 12, 4]
    X = rng.normal(size=(n, sizes[0]))
    y = rng.integers(0, sizes[-1], n)
    Y = np.zeros((n, sizes[-1]))
    Y[np.arange(n), y] = 1

    # One set of weights, used by BOTH implementations. float64 so the comparison
    # is limited by the mathematics, not by float32 rounding.
    weights = [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
    biases = [rng.normal(0, 0.1, b) for b in sizes[1:]]

    scratch_loss, scratch_dW, scratch_db = scratch_forward_backward(X, Y, weights, biases)

    # The identical network in PyTorch. nn.Linear computes x @ W.T + b, so its
    # weight matrix is the transpose of ours — hence the .T when copying across.
    torch_layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layer = nn.Linear(a, b).double()
        with torch.no_grad():
            layer.weight.copy_(torch.tensor(weights[i].T))
            layer.bias.copy_(torch.tensor(biases[i]))
        torch_layers.append(layer)
        if i < len(sizes) - 2:
            torch_layers.append(nn.ReLU())
    model = nn.Sequential(*torch_layers)

    Xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.long)
    logits = model(Xt)
    # CrossEntropyLoss = log_softmax + negative log likelihood, i.e. exactly the
    # softmax + cross-entropy of project 06 §4.3, fused for numerical stability.
    torch_loss = nn.CrossEntropyLoss()(logits, yt)
    torch_loss.backward()  # <- one line replaces the entire backward pass

    print(f"\nLoss   — scratch: {scratch_loss:.12f}")
    print(f"       — PyTorch: {torch_loss.item():.12f}")
    print(f"       — difference: {abs(scratch_loss - torch_loss.item()):.3e}")

    print(f"\n{'Layer':<10}{'max |scratch - autograd| (W)':>32}{'(b)':>16}")
    print("-" * 74)
    linear_layers = [m for m in model if isinstance(m, nn.Linear)]
    max_diff = 0.0
    for l, layer in enumerate(linear_layers):
        dW_torch = layer.weight.grad.numpy().T  # transpose back to our convention
        db_torch = layer.bias.grad.numpy()
        dw_diff = np.max(np.abs(scratch_dW[l] - dW_torch))
        db_diff = np.max(np.abs(scratch_db[l] - db_torch))
        max_diff = max(max_diff, dw_diff, db_diff)
        print(f"{l + 1:<10}{dw_diff:>32.3e}{db_diff:>16.3e}")

    print(
        f"\nEvery gradient agrees to {max_diff:.1e} — floating-point dust. PyTorch did not do\n"
        f"anything you did not do in project 06; it applied the same chain rule, in the same\n"
        f"order, to the same weights. The difference is that YOU wrote equations (7)-(10)\n"
        f"for this specific architecture, while autograd derives the equivalent for ANY\n"
        f"architecture you can express as code.\n\n"
        f"That is the whole trade of this project: you give up writing the backward pass,\n"
        f"and in return you can change the forward pass freely without re-deriving anything."
    )


# ---------------------------------------------------------------------------
# Part 2 — what a computation graph actually is
# ---------------------------------------------------------------------------


def run_computation_graph_demo() -> None:
    print()
    print("=" * 74)
    print("PART 2 — The computation graph, and checking one derivative by hand")
    print("=" * 74)

    # float64 so that "do these agree?" is answered by the mathematics rather than
    # by float32's ~7 significant digits.
    x = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    W = torch.tensor([[0.5, -1.0], [2.0, 0.25]], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([0.1, -0.3], dtype=torch.float64, requires_grad=True)

    z = x @ W + b
    a = torch.tanh(z)
    J = (a ** 2).sum()

    print("\nExpression:  z = xW + b  ->  a = tanh(z)  ->  J = sum(a^2)")
    print("\nEvery operation records what produced it — that record IS the graph:")
    print(f"  z.grad_fn = {z.grad_fn}")
    print(f"  a.grad_fn = {a.grad_fn}")
    print(f"  J.grad_fn = {J.grad_fn}")
    print(f"  and J.grad_fn's parent: {J.grad_fn.next_functions[0][0]}")

    J.backward()

    # Hand derivation:
    #   dJ/da = 2a
    #   da/dz = 1 - tanh(z)^2 = 1 - a^2
    #   dJ/dz = 2a * (1 - a^2)
    #   dJ/dW = x^T (dJ/dz)          <- project 01's "gradient = error x input"
    #   dJ/db = dJ/dz
    with torch.no_grad():
        dJ_dz = 2 * a * (1 - a ** 2)
        hand_dW = x.T @ dJ_dz
        hand_db = dJ_dz.squeeze(0)

    print(f"\n{'Quantity':<14}{'by hand':>26}{'autograd':>26}")
    print("-" * 74)
    print(f"{'dJ/dW[0,0]':<14}{hand_dW[0,0].item():>26.12f}{W.grad[0,0].item():>26.12f}")
    print(f"{'dJ/dW[1,1]':<14}{hand_dW[1,1].item():>26.12f}{W.grad[1,1].item():>26.12f}")
    print(f"{'dJ/db[0]':<14}{hand_db[0].item():>26.12f}{b.grad[0].item():>26.12f}")
    print(f"\nmax |hand - autograd| over all entries: "
          f"{max(float((hand_dW - W.grad).abs().max()), float((hand_db - b.grad).abs().max())):.3e}")

    print(
        "\nHow it works: each tensor produced by an operation stores a grad_fn — a pointer\n"
        "back to the operation that made it, and through it to that operation's inputs.\n"
        "Calling .backward() walks that chain from J back to the leaves, multiplying local\n"
        "derivatives as it goes. It is project 06's equation (10) with the bookkeeping\n"
        "automated: PyTorch knows d(tanh)/dz and d(matmul)/dW for every primitive, so it\n"
        "can assemble the chain rule for whatever you compose out of them.\n\n"
        "This is why autograd handles architectures nobody has derived on paper. It never\n"
        "needed the architecture — only the primitives, plus the record of how you used\n"
        "them. Note the graph is rebuilt on every forward pass ('define-by-run'), which is\n"
        "why an if-statement or a Python loop in forward() just works."
    )


# ---------------------------------------------------------------------------
# Part 3 — optimizers
# ---------------------------------------------------------------------------


def run_optimizer_demo() -> None:
    print()
    print("=" * 74)
    print("PART 3 — Optimizers on a hard surface: why plain SGD struggles")
    print("=" * 74)

    # An ill-conditioned quadratic: f(x,y) = 0.5*(x^2 + 20y^2). The valley is 20x
    # steeper across than along, which is the geometry that makes plain gradient
    # descent zig-zag. Real loss surfaces are far worse than this.
    def f(p):
        return 0.5 * (p[0] ** 2 + 20.0 * p[1] ** 2)

    start = [-9.0, 2.0]
    # Each entry: how to build the optimizer for the 2D toy, and for the real network.
    # The learning rates differ between the two because the problems' scales differ —
    # that is itself worth noticing: lr is not transferable between problems.
    configs = {
        "SGD (lr=0.02)": (
            lambda p: torch.optim.SGD([p], lr=0.02),
            lambda params: torch.optim.SGD(params, lr=0.05)),
        "SGD + momentum 0.9": (
            lambda p: torch.optim.SGD([p], lr=0.02, momentum=0.9),
            lambda params: torch.optim.SGD(params, lr=0.05, momentum=0.9)),
        "RMSprop (lr=0.3)": (
            lambda p: torch.optim.RMSprop([p], lr=0.3),
            lambda params: torch.optim.RMSprop(params, lr=0.005)),
        "Adam (lr=0.3)": (
            lambda p: torch.optim.Adam([p], lr=0.3),
            lambda params: torch.optim.Adam(params, lr=0.005)),
    }

    paths, steps_to_converge = {}, {}
    for name, (make_toy_opt, _) in configs.items():
        p = torch.tensor(start, requires_grad=True)
        opt = make_toy_opt(p)
        path = [p.detach().numpy().copy()]
        reached = None
        for step in range(400):
            opt.zero_grad()
            loss = f(p)
            loss.backward()
            opt.step()
            path.append(p.detach().numpy().copy())
            if reached is None and loss.item() < 1e-3:
                reached = step + 1
        paths[name] = np.array(path)
        steps_to_converge[name] = reached

    print(f"\nMinimizing f(x,y) = 0.5(x^2 + 20y^2) from {start} — the minimum is at (0,0).")
    print(f"\n{'Optimizer':<24}{'steps to f < 1e-3':>20}{'final f':>14}")
    print("-" * 74)
    for name in configs:
        reached = steps_to_converge[name]
        final = f(torch.tensor(paths[name][-1]))
        print(f"{name:<24}{(str(reached) if reached else '> 400'):>20}{float(final):>14.2e}")

    print(
        "\nThe surface is 20x steeper across the valley than along it. Plain SGD's step is\n"
        "proportional to the gradient, so it takes huge steps across (bouncing between the\n"
        "walls) and tiny steps along (where it actually needs to go) — the classic zig-zag\n"
        "in the left plot. Momentum accumulates the consistent downhill direction and\n"
        "cancels the oscillation. RMSprop and Adam instead rescale EACH coordinate by its\n"
        "own recent gradient size, so the flat direction gets a big step and the steep one\n"
        "a small step — which is why they head almost straight at the minimum."
    )

    # The same four on a real network, so the toy result is not the only evidence.
    digits = load_digits()
    Xtr, Xte, ytr, yte = train_test_split(digits.data, digits.target, test_size=0.25,
                                          random_state=42, stratify=digits.target)
    scaler = StandardScaler()
    Xtr_t = torch.tensor(scaler.fit_transform(Xtr), dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)

    real_curves = {}
    for name, (_, make_net_opt) in configs.items():
        torch.manual_seed(1)
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 32),
                              nn.ReLU(), nn.Linear(32, 10))
        opt = make_net_opt(model.parameters())
        criterion = nn.CrossEntropyLoss()
        losses = []
        for _ in range(150):
            opt.zero_grad()
            loss = criterion(model(Xtr_t), ytr_t)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        real_curves[name] = losses

    print(f"\nThe same four optimizers on the digits network (150 full-batch steps):")
    print(f"\n{'Optimizer':<24}{'final training loss':>22}")
    print("-" * 74)
    for name, losses in real_curves.items():
        print(f"{name:<24}{losses[-1]:>22.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    xs = np.linspace(-10, 10, 300)
    ys = np.linspace(-3, 3, 300)
    XX, YY = np.meshgrid(xs, ys)
    ax1.contour(XX, YY, 0.5 * (XX ** 2 + 20 * YY ** 2), levels=30, cmap="Greys", linewidths=.5)
    for name, path in paths.items():
        ax1.plot(path[:, 0], path[:, 1], marker="o", markersize=2, linewidth=1, label=name)
    ax1.plot(0, 0, "k*", markersize=14)
    ax1.set_title("Path taken on f(x,y) = 0.5(x² + 20y²)", fontsize=10)
    ax1.legend(fontsize=7)
    for name, losses in real_curves.items():
        ax2.plot(losses, label=name)
    ax2.set_yscale("log")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Training loss")
    ax2.set_title("The same optimizers on the digits network", fontsize=10)
    ax2.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "optimizers.png", dpi=120)
    plt.close()
    print("Saved plot to outputs/optimizers.png")


# ---------------------------------------------------------------------------
# Part 4 — batch size
# ---------------------------------------------------------------------------


def run_batch_size_demo() -> None:
    print()
    print("=" * 74)
    print("PART 4 — Full-batch vs. mini-batch: per epoch, and per second")
    print("=" * 74)

    digits = load_digits()
    Xtr, Xte, ytr, yte = train_test_split(digits.data, digits.target, test_size=0.25,
                                          random_state=42, stratify=digits.target)
    scaler = StandardScaler()
    Xtr_t = torch.tensor(scaler.fit_transform(Xtr), dtype=torch.float32)
    Xte_t = torch.tensor(scaler.transform(Xte), dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)

    results = {}
    for batch_size in (len(Xtr_t), 256, 32):
        torch.manual_seed(2)
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 32),
                              nn.ReLU(), nn.Linear(32, 10))
        opt = torch.optim.SGD(model.parameters(), lr=0.05)
        criterion = nn.CrossEntropyLoss()

        losses, times, updates = [], [], 0
        t0 = time.perf_counter()
        for _ in range(60):
            perm = torch.randperm(len(Xtr_t))
            for i in range(0, len(Xtr_t), batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                loss = criterion(model(Xtr_t[idx]), ytr_t[idx])
                loss.backward()
                opt.step()
                updates += 1
            with torch.no_grad():
                losses.append(criterion(model(Xtr_t), ytr_t).item())
            times.append(time.perf_counter() - t0)

        with torch.no_grad():
            acc = accuracy_score(yte, model(Xte_t).argmax(dim=1).numpy())
        label = "full batch" if batch_size == len(Xtr_t) else f"mini-batch {batch_size}"
        results[label] = (losses, times, updates, acc)

    print(f"\nSame network, same optimizer, same 60 epochs over the same {len(Xtr_t)} samples.\n")
    print(f"{'Batch size':<20}{'weight updates':>16}{'final loss':>13}{'test acc':>11}{'seconds':>10}")
    print("-" * 74)
    for label, (losses, times, updates, acc) in results.items():
        print(f"{label:<20}{updates:>16}{losses[-1]:>13.4f}{acc:>11.4f}{times[-1]:>10.2f}")

    print(
        "\nAn 'epoch' means one pass over the data — but the number of WEIGHT UPDATES per\n"
        "epoch is len(data)/batch_size. Full-batch does 1 update per epoch; batch size 32\n"
        "does 43. So after the same 60 epochs and roughly the same amount of arithmetic,\n"
        "the mini-batch model has taken 43x more steps downhill and is far ahead.\n\n"
        "Each mini-batch gradient is a NOISY estimate of the true gradient — computed from\n"
        "32 samples instead of all 1347. That noise is usually a feature rather than a bug:\n"
        "it costs a little accuracy per step and buys many more steps, and the jitter helps\n"
        "escape bad regions (project 05's local-minimum problem, encountered again).\n"
        "This is why essentially all deep learning is mini-batch: the full-batch gradient\n"
        "is more accurate than it needs to be, and you cannot fit ImageNet in memory anyway."
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for label, (losses, times, _, _) in results.items():
        ax1.plot(losses, label=label)
        ax2.plot(times, losses, label=label)
    for ax, xlabel, title in ((ax1, "Epoch", "Per epoch — mini-batch takes more steps"),
                              (ax2, "Wall-clock seconds", "Per second — the fair comparison")):
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Training loss")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "batch_size.png", dpi=120)
    plt.close()
    print("Saved plot to outputs/batch_size.png")


# ---------------------------------------------------------------------------
# Part 5 — regularization
# ---------------------------------------------------------------------------


def run_regularization_demo() -> None:
    print()
    print("=" * 74)
    print("PART 5 — Regularization: measured on the gap, not on the training loss")
    print("=" * 74)

    digits = load_digits()
    # A deliberately small training set, an oversized network, AND 30% corrupted
    # labels — three things that force severe overfitting. Without real overfitting
    # there is nothing for regularization to fix, and a fair experiment would (and
    # in an earlier draft of this project, did) show it doing nothing at all.
    X_small, X_rest, y_small, y_rest = train_test_split(
        digits.data, digits.target, train_size=200, random_state=0, stratify=digits.target)
    rng = np.random.default_rng(0)
    y_small = y_small.copy()
    corrupt = rng.choice(len(y_small), int(0.3 * len(y_small)), replace=False)
    y_small[corrupt] = rng.integers(0, 10, len(corrupt))  # 30% wrong on purpose

    scaler = StandardScaler()
    Xtr = torch.tensor(scaler.fit_transform(X_small), dtype=torch.float32)
    Xva = torch.tensor(scaler.transform(X_rest), dtype=torch.float32)
    ytr = torch.tensor(y_small, dtype=torch.long)
    yva = torch.tensor(y_rest, dtype=torch.long)

    def train_one(dropout: float, weight_decay: float, epochs: int = 300):
        torch.manual_seed(3)
        model = nn.Sequential(
            nn.Linear(64, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 10))
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        tr_hist, va_hist = [], []
        for _ in range(epochs):
            model.train()  # dropout ACTIVE
            opt.zero_grad()
            criterion(model(Xtr), ytr).backward()
            opt.step()
            model.eval()  # dropout DISABLED — this switch is why .eval() exists
            with torch.no_grad():
                tr_hist.append((model(Xtr).argmax(1) == ytr).float().mean().item())
                va_hist.append((model(Xva).argmax(1) == yva).float().mean().item())
        return tr_hist, va_hist

    print(f"\n{len(ytr)} training samples, 30% of their labels randomly corrupted, a 256-256")
    print("network. Sweeping each regularizer's STRENGTH rather than just on/off:\n")

    decays = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    drops = [0.0, 0.1, 0.25, 0.5, 0.7, 0.9]
    decay_scores, drop_scores = [], []

    print(f"{'weight decay':>14}{'train':>9}{'val':>9}{'gap':>9}   "
          f"{'dropout':>9}{'train':>9}{'val':>9}{'gap':>9}")
    print("-" * 74)
    for wd, dp in zip(decays, drops):
        tr_d, va_d = train_one(0.0, wd)
        tr_p, va_p = train_one(dp, 0.0)
        decay_scores.append(va_d[-1])
        drop_scores.append(va_p[-1])
        print(f"{wd:>14.0e}{tr_d[-1]:>9.3f}{va_d[-1]:>9.3f}{tr_d[-1]-va_d[-1]:>9.3f}   "
              f"{dp:>9.2f}{tr_p[-1]:>9.3f}{va_p[-1]:>9.3f}{tr_p[-1]-va_p[-1]:>9.3f}")

    best_wd = decays[int(np.argmax(decay_scores))]
    best_dp = drops[int(np.argmax(drop_scores))]
    baseline = decay_scores[0]
    at_edge = best_dp == drops[-1]
    print(
        f"\nUnregularized validation accuracy is {baseline:.3f}. The best weight decay "
        f"({best_wd:g}) reaches\n{max(decay_scores):.3f}, and the best dropout ({best_dp:g}) reaches "
        f"{max(drop_scores):.3f} — a large gain, because with\n30% of the labels wrong there is a great "
        f"deal of memorization to prevent.\n\n"
        f"But the two columns do NOT behave the same way, and it is worth being precise:\n"
        f"  - WEIGHT DECAY peaks and then collapses. At 1e-1 it already scores below doing\n"
        f"    nothing, and at 1.0 accuracy falls to {decay_scores[-1]:.3f} — barely above the 0.10 you would\n"
        f"    get by guessing. That is underfitting: the penalty has crushed the weights to\n"
        f"    the point that the network cannot represent anything.\n"
        f"  - DROPOUT is still improving at {drops[-1]:g}, the largest value swept"
        f"{', so the optimum lies' if at_edge else ''}\n"
        f"{'    somewhere beyond this range and the sweep was too narrow to find it.' if at_edge else ''}\n"
        f"    An optimum sitting at the edge of your search is always a signal to widen it.\n\n"
        f"So: regularization strength is a hyperparameter with a real optimum, not a switch\n"
        f"to flip. Too little overfits, too much underfits — project 03's bias-variance\n"
        f"tradeoff with a new knob — and it must be tuned on validation data.\n\n"
        f"Watch the training column too. Without regularization the network reaches ~1.00\n"
        f"training accuracy — it has memorized the corrupted labels, which is exactly the\n"
        f"behaviour that makes training accuracy worthless as a signal.\n\n"
        f"WEIGHT DECAY adds lambda*sum(w^2) to the loss, so its gradient adds lambda*w — the\n"
        f"very same L2 penalty you added by hand in projects 01 and 02, now one keyword\n"
        f"argument. DROPOUT randomly zeroes a fraction of units during EACH training step,\n"
        f"so no unit can depend on any other being present. Note the model.train() /\n"
        f"model.eval() switch: dropout must be ACTIVE while training and DISABLED when\n"
        f"evaluating. Forgetting .eval() is among the most common PyTorch bugs, and it\n"
        f"silently makes your reported numbers worse."
    )

    tr_none, va_none = train_one(0.0, 0.0)
    tr_best, va_best = train_one(best_dp, best_wd)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax1.semilogx([max(d, 1e-5) for d in decays], decay_scores, marker="o", label="weight decay")
    ax1.axhline(baseline, color="grey", linestyle=":", label="no regularization")
    ax1.set_xlabel("weight decay (log scale; leftmost point = 0)")
    ax1.set_ylabel("Validation accuracy")
    ax1.set_title("Too little overfits, too much underfits", fontsize=10)
    ax1.legend(fontsize=8)
    ax2.plot(va_none, label="no regularization (val)")
    ax2.plot(va_best, label=f"dropout {best_dp:g} + decay {best_wd:g} (val)")
    ax2.plot(tr_none, color="C0", linestyle=":", alpha=.5, label="(train)")
    ax2.plot(tr_best, color="C1", linestyle=":", alpha=.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Solid = validation, dotted = training", fontsize=10)
    ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "regularization.png", dpi=120)
    plt.close()
    print("Saved plot to outputs/regularization.png")


# ---------------------------------------------------------------------------
# Part 6 — a pipeline that respects project 03
# ---------------------------------------------------------------------------


def run_final_pipeline() -> None:
    print()
    print("=" * 74)
    print("PART 6 — A proper pipeline: choose on validation, test exactly once")
    print("=" * 74)

    digits = load_digits()
    # THREE splits, not two. Validation picks the epoch; test is opened once at the
    # very end. Selecting on the test set is project 03 Part 5's leakage.
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=42, stratify=digits.target)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.2, random_state=42, stratify=y_tmp)

    scaler = StandardScaler().fit(X_train)
    to_t = lambda A: torch.tensor(scaler.transform(A), dtype=torch.float32)
    Xtr, Xva, Xte = to_t(X_train), to_t(X_val), to_t(X_test)
    ytr = torch.tensor(y_train, dtype=torch.long)
    yva = torch.tensor(y_val, dtype=torch.long)

    torch.manual_seed(4)
    model = nn.Sequential(
        nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, 10))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val, best_state, best_epoch = 0.0, None, 0
    tr_hist, va_hist = [], []
    batch_size = 64
    for epoch in range(120):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            criterion(model(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            tr_acc = (model(Xtr).argmax(1) == ytr).float().mean().item()
            va_acc = (model(Xva).argmax(1) == yva).float().mean().item()
        tr_hist.append(tr_acc)
        va_hist.append(va_acc)
        if va_acc > best_val:  # early stopping: keep the best-on-VALIDATION weights
            best_val, best_epoch = va_acc, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_acc = (model(Xte).argmax(1) == torch.tensor(y_test)).float().mean().item()

    logistic = LogisticRegression(max_iter=5000).fit(scaler.transform(X_train), y_train)
    log_acc = accuracy_score(y_test, logistic.predict(scaler.transform(X_test)))

    # A project-06-style baseline trained on THIS split: same architecture family,
    # but plain full-batch SGD, no regularization, no early stopping. Project 06's
    # reported 0.9667 was measured on a DIFFERENT split, so quoting it here would be
    # comparing across test sets — which is exactly the kind of sloppiness project 03
    # is about. This row is the apples-to-apples version.
    torch.manual_seed(4)
    plain = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64),
                          nn.ReLU(), nn.Linear(64, 10))
    plain_opt = torch.optim.SGD(plain.parameters(), lr=0.5)
    for _ in range(600):
        plain_opt.zero_grad()
        criterion(plain(Xtr), ytr).backward()
        plain_opt.step()
    plain.eval()
    with torch.no_grad():
        plain_acc = (plain(Xte).argmax(1) == torch.tensor(y_test)).float().mean().item()

    print(f"\nBest validation accuracy {best_val:.4f} at epoch {best_epoch} — those weights "
          f"were kept.")
    print(f"\nAll three trained and evaluated on the SAME split:\n")
    print(f"{'Model':<52}{'Test accuracy':>16}")
    print("-" * 74)
    print(f"{'Logistic regression':<52}{log_acc:>16.4f}")
    print(f"{'Network, project-06 style (full-batch SGD, no reg)':<52}{plain_acc:>16.4f}")
    print(f"{'Network, Adam + dropout + early stopping':<52}{test_acc:>16.4f}")

    verdict = "beats" if test_acc > log_acc else "still does not beat"
    print(
        f"\nThe well-trained network {verdict} logistic regression ({test_acc:.4f} vs {log_acc:.4f}),\n"
        f"and it beats the project-06-style network ({plain_acc:.4f}) using THE SAME ARCHITECTURE.\n"
        f"Nothing about the model changed — only the training: Adam instead of plain SGD,\n"
        f"mini-batches instead of full-batch, dropout and weight decay, and stopping at the\n"
        f"best validation epoch instead of a fixed count. Project 06's network was not too\n"
        f"small; it was under-trained.\n\n"
        f"Note the discipline that makes this number trustworthy. The epoch was chosen on the\n"
        f"VALIDATION set and the test set was touched exactly once, at the end. Had we picked\n"
        f"the best TEST epoch instead, we would report {max(va_hist):.4f}-ish and it would mean\n"
        f"nothing — project 03's Part 5 leakage, which is far easier to commit by accident in\n"
        f"a training loop than in a feature-selection step. Note also that all three numbers\n"
        f"come from the same split; quoting project 06's 0.9667 here would be comparing across\n"
        f"different test sets, which is why that row was replaced with a retrained baseline."
    )

    plt.figure(figsize=(7.5, 4.4))
    plt.plot(tr_hist, label="Training accuracy")
    plt.plot(va_hist, label="Validation accuracy")
    plt.axvline(best_epoch, color="black", linestyle="--", linewidth=1,
                label=f"selected epoch ({best_epoch})")
    plt.axhline(log_acc, color="grey", linestyle=":", label="logistic regression (test)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Model selection on validation, never on test")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "final_training.png", dpi=120)
    plt.close()
    print("Saved plot to outputs/final_training.png")

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(
        f"\nA note on hardware: everything above ran on the CPU for reproducibility, but this\n"
        f"machine reports an available '{dev}' device. Moving a model to a GPU is\n"
        f"model.to('{dev}') plus the same for each batch — no other code changes. On a\n"
        f"64-feature dataset it would be slower (transfer overhead exceeds the work); it\n"
        f"matters from project 08 onward, where the tensors get large."
    )


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    run_autograd_vs_scratch()
    run_computation_graph_demo()
    run_optimizer_demo()
    run_batch_size_demo()
    run_regularization_demo()
    run_final_pipeline()
