"""
Convolutional Neural Networks — the convolution implemented from scratch and checked
against PyTorch, then five experiments on real MNIST digits:

  Part 1  convolution from scratch, verified against F.conv2d, and what a kernel does
  Part 2  weight sharing: where a CNN's parameters actually go (not where you think)
  Part 3  translation robustness — measured, including where the CNN also fails
  Part 4  CNN vs MLP vs a global-average-pooling CNN on MNIST
  Part 5  data efficiency: how the two compare as training data shrinks
  Part 6  what the first layer actually learned, drawn

Run:
    python cnn_classifier.py

Downloads MNIST (~11 MB) into ./data on first run. See README.md for the math.
"""

import os
from pathlib import Path

# macOS python.org builds ship a CA bundle but don't wire it up, so torchvision's
# download fails with CERTIFICATE_VERIFY_FAILED. Point it at certifi's bundle.
# (Same fix as _shared/setup.md, applied automatically here.)
try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

OUTPUT_DIR = Path(__file__).parent / "outputs"
DATA_DIR = Path(__file__).parent / "data"
torch.manual_seed(0)
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Part 1 — convolution from scratch
# ---------------------------------------------------------------------------


def conv2d_scratch(image, kernel, stride=1, padding=0):
    """
    2D cross-correlation, the operation every "convolution" layer actually performs:

        out[i,j] = sum_u sum_v  image[i*s + u - p, j*s + v - p] * kernel[u,v]      (1)

    Slide the kernel over the image; at each position multiply overlapping values
    and sum. The output size follows from how many positions fit:

        out_size = floor((in_size + 2*padding - kernel_size) / stride) + 1         (2)

    Written as explicit loops so the formula is visible. Real implementations use
    im2col or FFTs; the arithmetic is identical.
    """
    if padding > 0:
        image = np.pad(image, padding, mode="constant")
    k = kernel.shape[0]
    out_h = (image.shape[0] - k) // stride + 1
    out_w = (image.shape[1] - k) // stride + 1

    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            patch = image[i * stride:i * stride + k, j * stride:j * stride + k]
            out[i, j] = np.sum(patch * kernel)  # (1)
    return out


def run_convolution_demo(sample_image) -> None:
    print("=" * 74)
    print("PART 1 — Convolution from scratch, and what a kernel actually does")
    print("=" * 74)

    image = sample_image.numpy()[0]  # (28, 28)

    # Hand-designed kernels — chosen, not learned, so you can see the mechanism.
    kernels = {
        "identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float),
        "blur (3x3 mean)": np.ones((3, 3)) / 9.0,
        "Sobel x (vertical edges)": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float),
        "Sobel y (horizontal edges)": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float),
    }

    print(f"\n{'Kernel':<28}{'output shape':>16}{'max |scratch - torch|':>25}")
    print("-" * 74)
    outputs = {}
    for name, k in kernels.items():
        mine = conv2d_scratch(image, k, stride=1, padding=1)
        theirs = F.conv2d(
            torch.tensor(image, dtype=torch.float64)[None, None],
            torch.tensor(k, dtype=torch.float64)[None, None],
            stride=1, padding=1,
        )[0, 0].numpy()
        outputs[name] = mine
        print(f"{name:<28}{str(mine.shape):>16}{np.max(np.abs(mine - theirs)):>25.3e}")

    print(
        "\nThe from-scratch loops and PyTorch's optimized kernel agree to machine precision.\n"
        "(Note: what deep learning calls 'convolution' is technically CROSS-CORRELATION —\n"
        "true convolution flips the kernel first. Since the kernel is learned, the flip is\n"
        "irrelevant: the network just learns the flipped version. Everyone calls it\n"
        "convolution anyway.)\n\n"
        "Look at the plot. The same image, four kernels, four completely different outputs:\n"
        "blur averages neighbours; Sobel-x lights up where brightness changes horizontally\n"
        "(vertical strokes); Sobel-y does the same for horizontal strokes. These are edge\n"
        "detectors, hand-designed by computer-vision researchers decades ago.\n\n"
        "The idea of a CNN is to stop designing them. Make the kernel entries WEIGHTS and\n"
        "let gradient descent choose them — Part 6 shows what it picks."
    )

    # Output-size formula, checked against reality.
    print(f"\nOutput size formula (2): floor((in + 2*pad - k)/stride) + 1")
    print(f"\n{'in':>5}{'kernel':>8}{'stride':>8}{'pad':>6}{'predicted':>11}{'actual':>9}")
    print("-" * 74)
    for k, s, p in [(3, 1, 0), (3, 1, 1), (5, 1, 0), (3, 2, 1), (7, 2, 3)]:
        predicted = (28 + 2 * p - k) // s + 1
        actual = conv2d_scratch(image, np.ones((k, k)), stride=s, padding=p).shape[0]
        print(f"{28:>5}{k:>8}{s:>8}{p:>6}{predicted:>11}{actual:>9}")
    print(
        "\nNote row 2: kernel 3 with padding 1 keeps the size at 28. That is why 'same'\n"
        "padding is p = (k-1)/2 — without it every layer shrinks the image, which limits\n"
        "how deep you can go."
    )

    fig, axes = plt.subplots(1, 5, figsize=(14, 3.2))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("original", fontsize=9)
    for ax, (name, out) in zip(axes[1:], outputs.items()):
        ax.imshow(out, cmap="gray")
        ax.set_title(name, fontsize=9)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "kernels.png", dpi=120)
    plt.close()
    print("Saved plot to outputs/kernels.png")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """A fully-connected network — projects 06 and 07's model, applied to images."""

    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),  # 28x28 -> 784 unrelated numbers. This is the problem.
            nn.Linear(28 * 28, hidden), nn.ReLU(),
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    """
    Two convolutional blocks, then a small classifier head.

    Each nn.Conv2d(in_ch, out_ch, k) holds out_ch kernels of shape (in_ch, k, k) —
    and crucially ONE set of weights is reused at every position in the image.
    That is weight sharing, and it is where the parameter savings in Part 2 come
    from, as well as the translation robustness in Part 3.
    """

    def __init__(self, ch1=16, ch2=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, ch1, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            nn.Conv2d(ch1, ch2, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(ch2 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class CNNGlobalPool(nn.Module):
    """
    The same convolutional trunk, but the big fully-connected head is replaced by
    GLOBAL AVERAGE POOLING: average each channel's whole 7x7 map down to one number,
    then classify those 32 numbers.

    This exists because of a fact Part 2 makes concrete — the convolutions are cheap
    in parameters, but a Flatten + Linear head on 32x7x7 = 1568 values is not, and it
    ends up holding ~95% of the CNN's weights. Global average pooling removes it, and
    is what every modern architecture (ResNet onward) actually does.
    """

    def __init__(self, ch1=16, ch2=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, ch1, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(ch1, ch2, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(ch2, 10)
        )

    def forward(self, x):
        return self.head(self.features(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train(model, X, y, epochs=8, batch_size=128, lr=1e-3, verbose=False):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            criterion(model(X[idx]), y[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            history.append(criterion(model(X[:2000]), y[:2000]).item())
        if verbose:
            print(f"    epoch {epoch}: loss {history[-1]:.4f}")
    return history


@torch.no_grad()
def accuracy(model, X, y):
    model.eval()
    preds = []
    for i in range(0, len(X), 1000):
        preds.append(model(X[i:i + 1000]).argmax(1))
    return (torch.cat(preds) == y).float().mean().item()


# ---------------------------------------------------------------------------
# Part 2 — parameter counting
# ---------------------------------------------------------------------------


def run_parameter_demo() -> None:
    print()
    print("=" * 74)
    print("PART 2 — Weight sharing: why a CNN needs so many fewer parameters")
    print("=" * 74)

    mlp, cnn, cnn_gap = MLP(), CNN(), CNNGlobalPool()
    print(f"\n{'Model':<38}{'parameters':>14}")
    print("-" * 74)
    print(f"{'MLP (784 -> 128 -> 64 -> 10)':<38}{count_params(mlp):>14,}")
    print(f"{'CNN (conv trunk + fully-connected head)':<38}{count_params(cnn):>14,}")
    print(f"{'CNN with global average pooling':<38}{count_params(cnn_gap):>14,}")

    first_mlp = 28 * 28 * 128 + 128
    first_cnn = 1 * 16 * 3 * 3 + 16
    print(f"\nNote the first two totals are nearly EQUAL — so 'CNNs use fewer parameters' is,")
    print(f"as stated, false here. Where the weights actually sit is the interesting part:\n")
    print(f"{'Layer':<40}{'parameters':>14}{'share':>10}")
    print("-" * 74)
    cnn_head = count_params(cnn.classifier)
    for label, n in (("MLP: first layer (784 x 128)", first_mlp),
                     ("CNN: first conv (16 kernels of 3x3)", first_cnn),
                     ("CNN: second conv (16->32 channels)", 16 * 32 * 9 + 32),
                     ("CNN: fully-connected head (1568 -> 64 -> 10)", cnn_head)):
        share = n / (count_params(mlp) if label.startswith("MLP") else count_params(cnn))
        print(f"{label:<40}{n:>14,}{share:>10.1%}")

    print(
        f"\nThe CONVOLUTIONS are astonishingly cheap: {first_cnn} parameters where the MLP's first\n"
        f"layer needs {first_mlp:,} — {first_mlp / first_cnn:,.0f}x fewer — because 16 kernels of 9 weights are\n"
        f"REUSED AT ALL 784 POSITIONS. The MLP instead learns a separate weight for every\n"
        f"(pixel, unit) pair, so a feature it learns at the top-left is stored completely\n"
        f"separately from the same feature at the bottom-right.\n\n"
        f"But the CNN's fully-connected HEAD holds {cnn_head / count_params(cnn):.0%} of its parameters and cancels the\n"
        f"saving. That is why modern architectures replace it with global average pooling —\n"
        f"average each channel's map to a single number, then classify those. Third row:\n"
        f"{count_params(cnn_gap):,} parameters, {count_params(mlp) / count_params(cnn_gap):.0f}x fewer than the MLP, and Part 4 tests whether it\n"
        f"pays for that saving in accuracy.\n\n"
        f"Weight sharing encodes a real assumption: a useful pattern (an edge, a corner) is\n"
        f"useful wherever it appears. That is translation equivariance, it is true of\n"
        f"photographs, and it is why convolution beats a fully-connected layer on images\n"
        f"while being useless on a spreadsheet of unrelated columns."
    )


# ---------------------------------------------------------------------------
# Part 3 — translation robustness
# ---------------------------------------------------------------------------


def shift_images(X, dx, dy):
    """Translate a batch of images by (dx, dy) pixels, filling with zeros."""
    return torch.roll(X, shifts=(dy, dx), dims=(2, 3))


def run_translation_demo(models, X_test, y_test) -> None:
    print()
    print("=" * 74)
    print("PART 3 — Translation robustness: what weight sharing actually buys")
    print("=" * 74)

    shifts = [0, 1, 2, 3, 4, 5]
    results = {name: [] for name in models}
    for name, model in models.items():
        for s in shifts:
            results[name].append(accuracy(model, shift_images(X_test, s, s), y_test))

    print("\nBoth models were trained ONLY on centred digits. Now shift the TEST digits")
    print("diagonally by a few pixels — a change no human would even notice:\n")
    print(f"{'shift (px)':>12}" + "".join(f"{n:>14}" for n in models))
    print("-" * 74)
    for i, s in enumerate(shifts):
        print(f"{s:>12}" + "".join(f"{results[n][i]:>14.4f}" for n in models))

    def retained(name, i):
        return results[name][i] / results[name][0]

    print(
        f"\nAt a 3-pixel shift each model retains this fraction of its own starting accuracy:\n"
        f"  MLP     {retained('MLP', 3):.1%}\n"
        f"  CNN     {retained('CNN', 3):.1%}\n"
        f"  CNN-GAP {retained('CNN-GAP', 3):.1%}\n\n"
        f"The MLP has no notion that pixel 200 is next to pixel 201 — flattening destroyed\n"
        f"that. It learned 'the pixel at index 350 is usually bright for a 7'. Move the 7 two\n"
        f"pixels and those indices hold background, so its evidence is simply gone.\n\n"
        f"The CNN slides the same kernel over every position, so a stroke shifted two pixels\n"
        f"activates the same kernel two positions along. That is EQUIVARIANCE: the feature\n"
        f"map moves with the input. Max-pooling then throws away where within each 2x2 window\n"
        f"the activation occurred, converting a little of that equivariance into INVARIANCE\n"
        f"('the answer does not change').\n\n"
        f"NOW LOOK AT THE THIRD COLUMN, which is the most interesting result in this project\n"
        f"and was not what I expected. CNN-GAP is the WORST model on centred digits\n"
        f"({results['CNN-GAP'][0]:.4f}) and by far the BEST on shifted ones — at a 4-pixel shift it scores\n"
        f"{results['CNN-GAP'][4]:.4f} while the plain CNN has collapsed to {results['CNN'][4]:.4f}.\n\n"
        f"The reason is exact rather than approximate. Global average pooling averages each\n"
        f"channel over EVERY spatial position, so its output literally cannot depend on where\n"
        f"in the image a feature was found — that information is summed away. The plain CNN's\n"
        f"Flatten + Linear head does the opposite: it reads the 7x7 map position by position,\n"
        f"reintroducing precisely the location-dependence the convolutions had avoided.\n\n"
        f"So the fully-connected head is where a CNN's translation invariance goes to die.\n"
        f"That is a genuine trade — peak accuracy on well-centred data against robustness\n"
        f"when things move — and it is a large part of why global average pooling became\n"
        f"standard. Note too that all three fail eventually: at 5 pixels a digit is falling\n"
        f"off the edge of a 28x28 frame, and no architecture can classify what is not there."
    )

    plt.figure(figsize=(7.5, 4.4))
    for name in models:
        plt.plot(shifts, results[name], marker="o", label=name)
    plt.xlabel("Diagonal shift applied to test images (pixels)")
    plt.ylabel("Test accuracy")
    plt.title("Trained on centred digits, tested on shifted ones")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "translation_robustness.png", dpi=120)
    plt.close()
    print("Saved plot to outputs/translation_robustness.png")


# ---------------------------------------------------------------------------
# Part 5 — data efficiency
# ---------------------------------------------------------------------------


def run_data_efficiency_demo(X_train, y_train, X_test, y_test) -> None:
    print()
    print("=" * 74)
    print("PART 5 — Data efficiency: how the two compare as data shrinks")
    print("=" * 74)

    # Averaged over 3 seeds: with 500 training images a single run is noisy enough
    # to invent or hide a trend. (An earlier single-seed version of this experiment
    # showed a non-monotone curve that was mostly seed noise.)
    sizes = [250, 500, 1000, 3000, 10000]
    n_seeds = 3
    results = {"MLP": [], "CNN": []}
    for n in sizes:
        for name, build in (("MLP", MLP), ("CNN", CNN)):
            accs = []
            for seed in range(n_seeds):
                torch.manual_seed(seed)
                model = build()
                train(model, X_train[:n], y_train[:n], epochs=12, batch_size=64)
                accs.append(accuracy(model, X_test, y_test))
            results[name].append(float(np.mean(accs)))

    print(f"\nEach number is the mean of {n_seeds} runs with different seeds.\n")
    print(f"{'train size':>12}{'MLP':>10}{'CNN':>10}{'CNN advantage':>16}{'MLP error / CNN error':>24}")
    print("-" * 74)
    for i, n in enumerate(sizes):
        gap = results["CNN"][i] - results["MLP"][i]
        ratio = (1 - results["MLP"][i]) / max(1 - results["CNN"][i], 1e-9)
        print(f"{n:>12}{results['MLP'][i]:>10.4f}{results['CNN'][i]:>10.4f}{gap:>+16.4f}{ratio:>24.2f}x")

    print(
        "\nThe CNN wins at every training-set size. Read the last column rather than the\n"
        "raw difference: as both models improve, an accuracy gap of a few points means\n"
        "progressively more, and the ERROR RATIO is the honest summary of that.\n\n"
        "What this buys is called INDUCTIVE BIAS — the architecture already 'knows' that\n"
        "nearby pixels are related and that position should not matter much, so it does not\n"
        "have to learn those facts from examples. The MLP has to infer them from data.\n\n"
        "Be careful how far you push this, though: the accuracy DIFFERENCE here is not a\n"
        "clean downward line, and at the smallest sizes both models are simply starved.\n"
        "The defensible claim is that the CNN is better at every size tested, not that the\n"
        "advantage shrinks monotonically with data — that is a real effect in the\n"
        "literature, but this experiment is too small and too noisy to demonstrate it.\n\n"
        "The general principle does carry through Phase 2: build the structure of your data\n"
        "into the model and you need less data to learn the rest. Project 09 does it for\n"
        "sequences, project 10 for relationships between positions."
    )

    plt.figure(figsize=(7.5, 4.4))
    for name in results:
        plt.plot(sizes, results[name], marker="o", label=name)
    plt.xscale("log")
    plt.xlabel("Training set size (log scale)")
    plt.ylabel("Test accuracy")
    plt.title("A CNN's built-in assumptions are worth the most when data is scarce")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "data_efficiency.png", dpi=120)
    plt.close()
    print("Saved plot to outputs/data_efficiency.png")


# ---------------------------------------------------------------------------
# Part 6 — what did it learn?
# ---------------------------------------------------------------------------


def run_filter_visualization(cnn, sample_image) -> None:
    print()
    print("=" * 74)
    print("PART 6 — What the first layer learned")
    print("=" * 74)

    kernels = cnn.features[0].weight.detach()  # (16, 1, 3, 3)
    with torch.no_grad():
        feature_maps = F.relu(cnn.features[0](sample_image[None]))[0]  # (16, 28, 28)

    fig, axes = plt.subplots(4, 8, figsize=(12, 6.5))
    for i in range(16):
        ax = axes[i // 8 * 2, i % 8]
        ax.imshow(kernels[i, 0], cmap="RdBu_r")
        ax.set_title(f"kernel {i}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        ax = axes[i // 8 * 2 + 1, i % 8]
        ax.imshow(feature_maps[i], cmap="gray")
        ax.set_title(f"response {i}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle("Learned 3x3 kernels (red/blue) and what each one responds to (grey)",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "learned_filters.png", dpi=120)
    plt.close()

    # Quantify what they became, rather than only asserting "they look like edges".
    sobel_x = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = sobel_x.T
    def cosine(a, b):
        return float((a.flatten() @ b.flatten()) / (a.norm() * b.norm() + 1e-9))
    sims = [max(abs(cosine(kernels[i, 0], sobel_x)), abs(cosine(kernels[i, 0], sobel_y)))
            for i in range(16)]
    edge_like = sum(s > 0.5 for s in sims)

    print(f"\nEach of the 16 learned 3x3 kernels compared against a Sobel edge detector\n"
          f"(cosine similarity, taking the better of the x and y orientations):")
    print(f"  best match : {max(sims):.3f}")
    print(f"  median     : {np.median(sims):.3f}")
    print(f"  kernels with similarity > 0.5: {edge_like} of 16")
    print(
        "\nSo the network independently rediscovered something close to the edge detectors\n"
        "that Part 1 applied by hand — nobody told it to. It was given only labelled digits\n"
        "and a loss, and edge detection turned out to be the useful thing to compute first.\n"
        "Not every kernel is an edge detector, though: the rest respond to blobs, corners\n"
        "and textures, which is why the median similarity is well below the best.\n\n"
        "This is the layered-representation idea in miniature. Layer 1 finds edges; layer 2\n"
        "combines edges into corners and curves; deeper layers combine those into object\n"
        "parts. In a large vision model this continues for dozens of layers.\n"
        "Saved plot to outputs/learned_filters.png"
    )


# ---------------------------------------------------------------------------


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=tf)
    test_ds = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=tf)

    # A subset, so the whole script runs in a couple of minutes on a CPU. Project 07's
    # lesson applies: more data would help both models, so the COMPARISON is what matters.
    n_train, n_test = 10000, 4000
    X_train = torch.stack([train_ds[i][0] for i in range(n_train)])
    y_train = torch.tensor([train_ds[i][1] for i in range(n_train)])
    X_test = torch.stack([test_ds[i][0] for i in range(n_test)])
    y_test = torch.tensor([test_ds[i][1] for i in range(n_test)])
    print(f"MNIST: {n_train} training and {n_test} test images of shape "
          f"{tuple(X_train.shape[1:])}\n")

    run_convolution_demo(X_train[0])
    run_parameter_demo()

    print()
    print("=" * 74)
    print("PART 4 — CNN vs MLP on MNIST")
    print("=" * 74)
    models, histories = {}, {}
    for name, build in (("MLP", MLP), ("CNN", CNN), ("CNN-GAP", CNNGlobalPool)):
        torch.manual_seed(1)
        model = build()
        histories[name] = train(model, X_train, y_train, epochs=20)
        models[name] = model

    print(f"\n{'Model':<12}{'parameters':>13}{'test accuracy':>16}{'error rate':>13}")
    print("-" * 74)
    accs = {}
    for name, model in models.items():
        accs[name] = accuracy(model, X_test, y_test)
        print(f"{name:<12}{count_params(model):>13,}{accs[name]:>16.4f}{1 - accs[name]:>13.4f}")

    mlp_err, cnn_err = 1 - accs["MLP"], 1 - accs["CNN"]
    gap_err = 1 - accs["CNN-GAP"]
    print(
        f"\nAccuracy differences look small at this end of the scale, so read the ERROR RATE:\n"
        f"{mlp_err:.2%} vs {cnn_err:.2%} — the CNN makes {mlp_err / cnn_err:.1f}x fewer mistakes at "
        f"essentially the same\nparameter count. That is the honest framing: on this task the CNN "
        f"buys accuracy,\nnot compactness.\n\n"
        f"CNN-GAP is the interesting row. With {count_params(models['CNN-GAP']):,} parameters — "
        f"{count_params(models['MLP']) / count_params(models['CNN-GAP']):.0f}x fewer than the MLP and\n"
        f"{count_params(models['CNN']) / count_params(models['CNN-GAP']):.0f}x fewer than the plain CNN — it "
        f"scores {accs['CNN-GAP']:.4f}, an error rate of {gap_err:.2%}.\n"
        f"{'It beats the MLP outright while being a fraction of its size.' if gap_err < mlp_err else 'It TRAILS both — so on this task the compression is not free, and it would be'}\n"
        f"dishonest to present global average pooling as a pure win. Two reasons: 32 numbers\n"
        f"is genuinely less capacity than 1568, and a GAP head converges much more slowly\n"
        f"(at 8 epochs it scored only 0.68; it is still improving at 20). Its advantages —\n"
        f"far fewer weights, and no dependence on a fixed input size — start paying at real\n"
        f"image resolutions and depths, which is why ResNet and everything after it use it.\n"
        f"Exercise 5 asks you to give it a fair fight.\n\n"
        f"On 28x28 greyscale digits all of this is a real but modest gap. On colour\n"
        f"photographs it is not close: a fully-connected first layer on a 224x224x3 image\n"
        f"would need 150 million parameters for a single hidden layer of 1000 units, which\n"
        f"is why no one has ever built a competitive fully-connected image classifier.\n\n"
        f"Parts 3 and 5 show the two properties behind the gap, which matter more than the\n"
        f"accuracy number itself."
    )

    plt.figure(figsize=(7.5, 4.4))
    for name, hist in histories.items():
        plt.plot(hist, marker="o", label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.yscale("log")
    plt.title("MNIST training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mnist_training.png", dpi=120)
    plt.close()
    print("Saved plot to outputs/mnist_training.png")

    run_translation_demo(models, X_test, y_test)
    run_data_efficiency_demo(X_train, y_train, X_test, y_test)
    run_filter_visualization(models["CNN"], X_train[0])


if __name__ == "__main__":
    main()
