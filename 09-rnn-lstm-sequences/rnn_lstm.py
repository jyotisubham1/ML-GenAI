"""
Recurrent networks — an RNN cell from scratch, the vanishing-gradient problem
measured rather than described, and the LSTM derived as the specific fix.

  Part 1  an RNN cell from scratch, verified against nn.RNN
  Part 2  gradients through time, measured per timestep: RNN vs LSTM
  Part 3  the memory task — recall the first token after N steps
  Part 4  exploding gradients, and what clipping actually does
  Part 5  inside the LSTM: watching the forget gate hold information
  Part 6  a character-level language model, scored by perplexity

Run:
    python rnn_lstm.py

See README.md for the math behind every formula referenced in the comments below.
"""

import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

OUTPUT_DIR = Path(__file__).parent / "outputs"
torch.manual_seed(0)
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Part 1 — an RNN cell from scratch
# ---------------------------------------------------------------------------


def rnn_forward_scratch(X, W_xh, W_hh, b_h, h0):
    """
    The entire recurrent network:

        h_t = tanh(x_t W_xh + h_{t-1} W_hh + b_h)                              (1)

    One equation, applied in a loop. The SAME weights are used at every timestep —
    that is weight sharing again, exactly as in project 08, but across TIME instead
    of across space. It is what lets one model handle any sequence length.

    X: (batch, seq_len, input_size). Returns all hidden states and the final one.
    """
    batch, seq_len, _ = X.shape
    h = h0
    hidden_states = []
    for t in range(seq_len):
        h = np.tanh(X[:, t, :] @ W_xh + h @ W_hh + b_h)  # (1)
        hidden_states.append(h.copy())
    return np.stack(hidden_states, axis=1), h


def run_scratch_rnn_demo() -> None:
    print("=" * 74)
    print("PART 1 — An RNN cell from scratch, checked against nn.RNN")
    print("=" * 74)

    batch, seq_len, input_size, hidden_size = 4, 7, 5, 6
    rng = np.random.default_rng(1)
    X = rng.normal(size=(batch, seq_len, input_size))

    torch_rnn = nn.RNN(input_size, hidden_size, batch_first=True, nonlinearity="tanh").double()

    # PyTorch stores weights as (hidden, input) and applies x @ W.T, so transpose to
    # get our (input, hidden) convention. It also keeps two bias vectors (one for the
    # input term, one for the hidden term); they are always added together, so their
    # sum is our single b_h.
    p = dict(torch_rnn.named_parameters())
    W_xh = p["weight_ih_l0"].detach().numpy().T
    W_hh = p["weight_hh_l0"].detach().numpy().T
    b_h = (p["bias_ih_l0"] + p["bias_hh_l0"]).detach().numpy()

    h0 = np.zeros((batch, hidden_size))
    mine, mine_last = rnn_forward_scratch(X, W_xh, W_hh, b_h, h0)
    theirs, theirs_last = torch_rnn(torch.tensor(X))

    print(f"\nSequence of {seq_len} steps, batch {batch}, hidden size {hidden_size}.")
    print(f"  max |scratch - nn.RNN| over all hidden states: "
          f"{np.max(np.abs(mine - theirs.detach().numpy())):.3e}")
    print(f"  max |scratch - nn.RNN| at the final step:      "
          f"{np.max(np.abs(mine_last - theirs_last.detach().numpy()[0])):.3e}")

    print(
        "\nThat is the whole model: h_t = tanh(x_t W_xh + h_{t-1} W_hh + b). The hidden\n"
        "state h is a fixed-size summary of everything seen so far, rewritten at every\n"
        "step. Notice what makes it different from projects 06-08: the output of the layer\n"
        "is fed back INTO the layer. That loop is the only new idea here.\n\n"
        "Because the same W_hh is applied at every step, one model handles a sequence of\n"
        "any length — which a fully-connected network cannot do, since its input size is\n"
        "fixed. That is the sequence analogue of project 08's weight sharing across space."
    )


# ---------------------------------------------------------------------------
# Part 2 — gradients through time
# ---------------------------------------------------------------------------


def gradient_flow_of(model, seq_len: int, input_size: int = 1):
    """
    How much does the loss at the END of a sequence depend on the input at step t?

    Attach a gradient to every input step, run the sequence, take a loss on the FINAL
    output only, and read off |dL/dx_t| for each t. That number IS the strength of the
    learning signal reaching step t. See README §4.3.
    """
    x = torch.randn(1, seq_len, input_size, requires_grad=True)
    out, _ = model(x)
    out[:, -1, :].sum().backward()
    return x.grad.abs().sum(dim=2).squeeze().detach().numpy()


def train_lstm_on_memory(seq_len, hidden=32, epochs=25):
    """An LSTM actually trained to remember, so its gates are no longer at their defaults."""
    X, y = make_memory_task(2000, seq_len)
    torch.manual_seed(0)
    model = Recaller("LSTM", hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            crit(model(X[idx]), y[idx]).backward()
            opt.step()
    return model


def run_gradient_flow_demo() -> None:
    print()
    print("=" * 74)
    print("PART 2 — Measuring the vanishing gradient through time")
    print("=" * 74)

    seq_len, hidden = 60, 32
    curves = {}

    torch.manual_seed(0)
    curves["RNN (untrained)"] = gradient_flow_of(nn.RNN(1, hidden, batch_first=True), seq_len)
    torch.manual_seed(0)
    curves["LSTM (untrained)"] = gradient_flow_of(nn.LSTM(1, hidden, batch_first=True), seq_len)

    # The standard trick: initialize the forget gate's bias positive, so sigmoid(b) is
    # near 1 and the cell state is preserved by default rather than halved.
    torch.manual_seed(0)
    lstm_fb = nn.LSTM(1, hidden, batch_first=True)
    with torch.no_grad():
        lstm_fb.bias_ih_l0[hidden:2 * hidden].fill_(3.0)  # gates packed as i, f, g, o
    curves["LSTM (forget bias = 3)"] = gradient_flow_of(lstm_fb, seq_len)

    # The same measurement through the RECURRENT weights of an LSTM that has actually
    # been trained on a task requiring memory (its input size is the embedding's 16).
    trained = train_lstm_on_memory(seq_len)
    curves["LSTM (trained to remember)"] = gradient_flow_of(trained.rnn, seq_len, input_size=16)

    print(f"\n|dL/dx_t| — how strongly the loss at step {seq_len} depends on the input at step t.\n")
    print(f"{'steps back':>11}" + "".join(f"{n:>26}" for n in list(curves)[:2]))
    print("-" * 74)
    for back in (0, 10, 20, 40, 59):
        t = seq_len - 1 - back
        print(f"{back:>11}" + "".join(f"{curves[n][t]:>26.3e}" for n in list(curves)[:2]))

    print(f"\n{'steps back':>11}" + "".join(f"{n:>26}" for n in list(curves)[2:]))
    print("-" * 74)
    for back in (0, 10, 20, 40, 59):
        t = seq_len - 1 - back
        print(f"{back:>11}" + "".join(f"{curves[n][t]:>26.3e}" for n in list(curves)[2:]))

    print(f"\n{'Model':<30}{'gradient at t=1 / at t=60':>28}")
    print("-" * 74)
    for name, g in curves.items():
        print(f"{name:<30}{g[0] / max(g[-1], 1e-30):>28.2e}")

    print(
        "\nThe RNN decays exponentially — a straight line on a log plot is what exponential\n"
        "decay looks like. By 40 steps back the signal is ~1e-9 of its value at the end, so\n"
        "the weights receive essentially no information about what happened there. An RNN\n"
        "does not 'forget slowly'; the gradient that would teach it to remember never\n"
        "arrives.\n\n"
        "Why worse than project 06's deep network: there, each layer multiplied by a\n"
        "DIFFERENT matrix. Here the SAME W_hh is applied at every step, so the product\n"
        "becomes a pure exponential in its largest eigenvalue — below 1 it vanishes, above 1\n"
        "it explodes (Part 4). There is no safe middle for long sequences.\n\n"
        "NOW THE PART THAT IS USUALLY GLOSSED OVER. An UNTRAINED LSTM decays just as badly\n"
        "as the RNN — look at the first table. Textbooks often say 'the LSTM solves the\n"
        "vanishing gradient', full stop, and measured at initialization that is simply false.\n\n"
        "The reason is in the mechanism. The cell update is c_t = f_t * c_(t-1) + i_t * g_t,\n"
        "so d c_t / d c_(t-1) = f_t. At initialization the forget gate's bias is 0, so\n"
        "f = sigmoid(0) = 0.5 and the cell state is HALVED every step: 0.5^60 = 1e-18. The\n"
        "LSTM has an additive path, but it starts with the valve half-closed.\n\n"
        "The last two rows show what actually fixes it. Setting the forget-gate bias to 3\n"
        "makes f = sigmoid(3) = 0.95 from the start, and the decay improves by orders of\n"
        "magnitude — this is why 'initialize the forget bias to 1 or 2' is standard advice.\n"
        "Training the LSTM on a task that REQUIRES memory does the same thing by learning:\n"
        "Part 5 opens the model up and shows its learned forget gate sitting near 1.\n\n"
        "So the honest claim is not 'LSTMs do not have vanishing gradients'. It is: an LSTM\n"
        "CAN keep gradients alive over long spans, because a learnable gate controls the\n"
        "decay rate, whereas a plain RNN's decay is fixed by its weight matrix and cannot be\n"
        "chosen per timestep."
    )

    plt.figure(figsize=(8, 4.6))
    for name, g in curves.items():
        plt.plot(range(seq_len - 1, -1, -1), g, label=name)
    plt.yscale("log")
    plt.xlabel("Steps back in time from the loss")
    plt.ylabel("|dL/dx_t|  (log scale)")
    plt.title("How far back the learning signal actually reaches")
    plt.legend(fontsize=8)
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gradient_flow.png", dpi=120)
    plt.close()
    print("Saved plot to outputs/gradient_flow.png")


# ---------------------------------------------------------------------------
# Part 3 — the memory task
# ---------------------------------------------------------------------------


def make_memory_task(n_samples, seq_len, n_symbols=8):
    """
    Sequence: [signal, noise, noise, ..., noise]. Target: reproduce `signal`.

    Everything after the first position is uninformative, so the model can only
    succeed by CARRYING the first token across the whole sequence. Accuracy on this
    task is therefore a direct measurement of memory span.
    """
    X = RNG.integers(1, n_symbols, size=(n_samples, seq_len))
    y = X[:, 0].copy()
    X[:, 1:] = 0  # everything after step 0 carries no information at all
    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class Recaller(nn.Module):
    def __init__(self, cell, n_symbols=8, hidden=48, forget_bias=None):
        super().__init__()
        self.embed = nn.Embedding(n_symbols, 16)
        base = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[cell]
        self.rnn = base(16, hidden, batch_first=True)
        if forget_bias is not None:
            # Gates are packed as i, f, g, o — the second block is the forget gate.
            with torch.no_grad():
                self.rnn.bias_ih_l0[hidden:2 * hidden].fill_(forget_bias)
        self.out = nn.Linear(hidden, n_symbols)

    def forward(self, x):
        h, _ = self.rnn(self.embed(x))
        return self.out(h[:, -1, :])  # classify from the FINAL hidden state


def run_memory_task_demo() -> None:
    print()
    print("=" * 74)
    print("PART 3 — The memory task: how far back can each architecture remember?")
    print("=" * 74)

    lengths = [5, 10, 20, 40, 80]
    # Part 2 predicted that the default LSTM would struggle and that a positive
    # forget-gate bias would fix it. This is where that prediction gets tested.
    cells = ["RNN", "LSTM", "LSTM fb=3", "GRU"]
    n_seeds = 3
    results = {c: [] for c in cells}
    spreads = {c: [] for c in cells}

    for L in lengths:
        X, y = make_memory_task(2000, L)
        Xte, yte = make_memory_task(500, L)
        for cell in cells:
            accs = []
            for seed in range(n_seeds):
                torch.manual_seed(seed)
                model = (Recaller("LSTM", forget_bias=3.0) if cell == "LSTM fb=3"
                         else Recaller(cell))
                opt = torch.optim.Adam(model.parameters(), lr=3e-3)
                crit = nn.CrossEntropyLoss()
                for _ in range(30):
                    perm = torch.randperm(len(X))
                    for i in range(0, len(X), 64):
                        idx = perm[i:i + 64]
                        opt.zero_grad()
                        crit(model(X[idx]), y[idx]).backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        opt.step()
                with torch.no_grad():
                    accs.append((model(Xte).argmax(1) == yte).float().mean().item())
            results[cell].append(float(np.mean(accs)))
            spreads[cell].append((min(accs), max(accs)))

    print(f"\nRecall the first symbol after N uninformative steps (7 classes, chance = 0.143).")
    print(f"Mean over {n_seeds} seeds, with (min-max) beneath — these runs are genuinely noisy:\n")
    print(f"{'seq length':>12}" + "".join(f"{c:>13}" for c in cells))
    print("-" * 74)
    for i, L in enumerate(lengths):
        print(f"{L:>12}" + "".join(f"{results[c][i]:>13.3f}" for c in cells))
        print(f"{'':>12}" + "".join(f"{f'({spreads[c][i][0]:.2f}-{spreads[c][i][1]:.2f})':>13}"
                                    for c in cells))

    print(
        "\nEverything is equivalent at 5 steps — the task is trivial when the signal is close.\n"
        "The architectures separate as the gap grows, and chance is 0.143 (7 symbols), so\n"
        "read each column for where it falls back to that line. That length is the model's\n"
        "practical memory span. Nothing about the task got harder: the information is still\n"
        "sitting at step 0 in plain sight. What changed is whether a gradient can reach back\n"
        "far enough to teach the model to keep it.\n\n"
        "THIS TABLE IS PART 2'S PREDICTION, TESTED. Part 2 measured gradient flow and found\n"
        "the default LSTM decaying as badly as the RNN, with a positive forget-gate bias\n"
        "fixing it by twelve orders of magnitude. If that measurement meant anything, the\n"
        "default LSTM should fail here and the fb=3 version should succeed. It does:\n"
        "  - the plain RNN is unreliable by 10 steps (note the 0.29-1.00 spread across\n"
        "    seeds) and is at chance by 40;\n"
        "  - the DEFAULT LSTM is no better — it fails at 20, worse than the RNN, which is\n"
        "    not what the textbook summary of LSTMs would lead you to expect;\n"
        "  - LSTM fb=3 and the GRU both solve 40 steps perfectly, every seed.\n\n"
        "So 'LSTMs handle long dependencies' is really 'LSTMs CAN, if their forget gate is\n"
        "initialized or trained to stay open'. The GRU gets there without the trick because\n"
        "its update gate couples remembering and forgetting into one term, which happens to\n"
        "start in a friendlier place.\n\n"
        "At 80 steps everything collapses except a single lucky fb=3 seed (0.13-0.83 — read\n"
        "the spread, not the mean). Gating buys perhaps an order of magnitude in span; it\n"
        "does not buy unlimited memory. That limit is what project 10 removes."
    )

    plt.figure(figsize=(7.5, 4.4))
    for c in cells:
        plt.plot(lengths, results[c], marker="o", label=c)
    plt.axhline(1 / 7, color="grey", linestyle=":", label="chance (1/7)")
    plt.xlabel("Sequence length (steps between the signal and the answer)")
    plt.ylabel("Test accuracy")
    plt.title("How far back can it remember?")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "memory_task.png", dpi=120)
    plt.close()
    print("Saved plot to outputs/memory_task.png")


# ---------------------------------------------------------------------------
# Part 4 — exploding gradients and clipping
# ---------------------------------------------------------------------------


def run_exploding_gradient_demo() -> None:
    print()
    print("=" * 74)
    print("PART 4 — The other failure: exploding gradients, and what clipping does")
    print("=" * 74)

    print("\nThe same RNN, with W_hh scaled to different spectral radii. Theory (README")
    print("§4.3) says the gradient over T steps behaves like (largest eigenvalue)^T:\n")
    print(f"{'spectral radius':>17}{'||dL/dx|| at t=1':>22}{'behaviour':>20}")
    print("-" * 74)

    seq_len, hidden = 50, 32
    for scale in (0.5, 0.9, 1.0, 1.1, 1.5):
        torch.manual_seed(0)
        rnn = nn.RNN(1, hidden, batch_first=True)
        with torch.no_grad():
            W = rnn.weight_hh_l0
            radius = torch.linalg.eigvals(W).abs().max().real
            W *= scale / radius  # rescale so the largest eigenvalue is exactly `scale`
        x = torch.randn(1, seq_len, 1, requires_grad=True)
        out, _ = rnn(x)
        out[:, -1, :].sum().backward()
        g = x.grad.abs().squeeze()[0].item()
        label = "vanishes" if g < 1e-6 else ("explodes" if g > 1e3 else "usable")
        print(f"{scale:>17.1f}{g:>22.3e}{label:>20}")

    print(
        "\nThe knife edge is real: below 1 the signal dies, above 1 it blows up, and only a\n"
        "narrow band is trainable. tanh's derivative (at most 1, usually less) pulls the\n"
        "effective factor down further, which is why plain RNNs vanish far more often than\n"
        "they explode.\n\n"
        "Exploding gradients have a blunt but effective fix — GRADIENT CLIPPING. If the\n"
        "gradient vector is longer than a threshold, rescale it to that length, keeping its\n"
        "direction:"
    )

    # Clipping, demonstrated on a run that genuinely explodes. To get there we need
    # the regime the table above identifies as dangerous: a long sequence AND recurrent
    # weights with spectral radius above 1. (An earlier version of this demo used a
    # short sequence and default weights, where the gradient never exceeded 0.7 — so
    # clipping never triggered and the two runs were identical. If your demonstration
    # of a fix shows no difference, check that the problem is actually present.)
    X, y = make_memory_task(600, 30)
    print()
    for clip in (None, 1.0):
        torch.manual_seed(2)
        model = Recaller("RNN")
        with torch.no_grad():
            W = model.rnn.weight_hh_l0
            W *= 2.5 / torch.linalg.eigvals(W).abs().max().real
        opt = torch.optim.SGD(model.parameters(), lr=0.5)
        crit = nn.CrossEntropyLoss()
        losses, norms = [], []
        for _ in range(150):
            opt.zero_grad()
            loss = crit(model(X), y)
            loss.backward()
            norms.append(torch.sqrt(sum((p.grad ** 2).sum()
                                        for p in model.parameters())).item())
            if clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            losses.append(loss.item())
        tag = "no clipping" if clip is None else f"clip at {clip}"
        finite = [n for n in norms if math.isfinite(n)]
        end = "diverged to nan" if not math.isfinite(losses[-1]) else f"{losses[-1]:.4f}"
        print(f"  {tag:<14} final loss {end:>16}   largest gradient norm "
              f"{(max(finite) if finite else float('nan')):.2e}")

    print(
        "\n    clip:  if ||g|| > threshold:  g <- threshold * g / ||g||\n\n"
        "Clipping caps the step SIZE while preserving its DIRECTION, so one unlucky batch\n"
        "cannot destroy the weights. It is a hack rather than a principled fix — it does\n"
        "nothing for vanishing gradients, which need an architectural change. That is the\n"
        "LSTM."
    )


# ---------------------------------------------------------------------------
# Part 5 — inside the LSTM
# ---------------------------------------------------------------------------


def run_lstm_gates_demo() -> None:
    print()
    print("=" * 74)
    print("PART 5 — Inside a trained LSTM: watching the forget gate hold on")
    print("=" * 74)

    seq_len = 40
    X, y = make_memory_task(3000, seq_len)
    # Use the forget-bias init, because Parts 2 and 3 showed the default LSTM simply
    # fails at 40 steps — and there is nothing to learn from inspecting the gates of a
    # model that never solved the task.
    torch.manual_seed(3)
    model = Recaller("LSTM", forget_bias=3.0)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    crit = nn.CrossEntropyLoss()
    for _ in range(40):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            crit(model(X[idx]), y[idx]).backward()
            opt.step()

    Xte, yte = make_memory_task(200, seq_len)
    with torch.no_grad():
        acc = (model(Xte).argmax(1) == yte).float().mean().item()
    print(f"\nTrained LSTM on the {seq_len}-step memory task: test accuracy {acc:.3f}")

    # Re-run the LSTM cell by hand so the gates are visible. PyTorch packs the four
    # gates into one matrix, in the order i, f, g, o.
    lstm = model.rnn
    W_ih, W_hh = lstm.weight_ih_l0, lstm.weight_hh_l0
    b = lstm.bias_ih_l0 + lstm.bias_hh_l0
    H = lstm.hidden_size

    with torch.no_grad():
        emb = model.embed(Xte[:64])
        h = torch.zeros(64, H)
        c = torch.zeros(64, H)
        forget_means, cell_norms = [], []
        for t in range(seq_len):
            gates = emb[:, t, :] @ W_ih.T + h @ W_hh.T + b
            i_g, f_g, g_g, o_g = gates.chunk(4, dim=1)
            i_g, f_g, o_g = torch.sigmoid(i_g), torch.sigmoid(f_g), torch.sigmoid(o_g)
            g_g = torch.tanh(g_g)
            c = f_g * c + i_g * g_g  # (5) the additive cell update
            h = o_g * torch.tanh(c)
            forget_means.append(f_g.mean().item())
            cell_norms.append(c.norm(dim=1).mean().item())

    print(f"\n{'step':>6}{'mean forget gate':>20}{'mean |cell state|':>20}")
    print("-" * 74)
    for t in (0, 1, 2, 5, 10, 20, 39):
        print(f"{t:>6}{forget_means[t]:>20.4f}{cell_norms[t]:>20.4f}")

    print(
        f"\nThe forget gate averages {np.mean(forget_means[1:]):.3f} across the uninformative steps — "
        f"close to 1,\nmeaning 'keep everything'. That is the network having LEARNED to hold the "
        f"cell\nstate steady, because on this task the only useful strategy is to memorize step 0\n"
        f"and ignore the rest.\n\n"
        f"Why this matters for gradients: the cell update is c_t = f_t * c_(t-1) + i_t * g_t,\n"
        f"so d c_t / d c_(t-1) = f_t. When f_t is near 1, the gradient flowing back through\n"
        f"time is multiplied by roughly 1 at each step instead of by a matrix — it neither\n"
        f"vanishes nor explodes. The forget gate IS the mechanism, and it is learned per\n"
        f"timestep rather than fixed, so the model can also choose to forget when that is\n"
        f"the right thing to do."
    )

    plt.figure(figsize=(7.5, 4.4))
    plt.plot(forget_means, label="mean forget gate f_t")
    plt.plot(np.array(cell_norms) / max(cell_norms), label="mean |c_t| (normalized)")
    plt.axhline(1.0, color="grey", linestyle=":")
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.ylim(0, 1.15)
    plt.title("A trained LSTM holds its forget gate open to preserve the cell state")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lstm_gates.png", dpi=120)
    plt.close()
    print("Saved plot to outputs/lstm_gates.png")


# ---------------------------------------------------------------------------
# Part 6 — a character-level language model
# ---------------------------------------------------------------------------

GRAMMAR_ADJ = ["quiet", "golden", "restless", "hollow", "bright", "distant"]
GRAMMAR_NOUN = ["river", "sparrow", "mountain", "lantern", "harbour", "meadow"]
GRAMMAR_VERB = ["carries", "remembers", "hides", "follows", "shelters", "answers"]


def make_corpus(n_sentences=1200):
    """
    A corpus from a fixed grammar: "the ADJ NOUN VERB the ADJ NOUN ."

    Synthetic on purpose. Real text needs a much larger model and corpus to produce
    anything, whereas here the rules are known exactly, so "did the model learn the
    structure?" has a checkable answer rather than a vibe.
    """
    rng = np.random.default_rng(7)
    out = []
    for _ in range(n_sentences):
        out.append(
            f"the {rng.choice(GRAMMAR_ADJ)} {rng.choice(GRAMMAR_NOUN)} "
            f"{rng.choice(GRAMMAR_VERB)} the {rng.choice(GRAMMAR_ADJ)} "
            f"{rng.choice(GRAMMAR_NOUN)} .\n"
        )
    return "".join(out)


class CharModel(nn.Module):
    def __init__(self, cell, vocab, hidden=128):
        super().__init__()
        self.embed = nn.Embedding(vocab, 32)
        self.rnn = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[cell](
            32, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab)

    def forward(self, x, state=None):
        h, state = self.rnn(self.embed(x), state)
        return self.out(h), state


def run_language_model_demo() -> None:
    print()
    print("=" * 74)
    print("PART 6 — A character-level language model, scored by perplexity")
    print("=" * 74)

    text = make_corpus()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]
    seq_len = 64

    def batches(d, batch_size=64):
        ix = torch.randint(0, len(d) - seq_len - 1, (batch_size,))
        x = torch.stack([d[i:i + seq_len] for i in ix])
        y = torch.stack([d[i + 1:i + seq_len + 1] for i in ix])
        return x, y

    print(f"\nCorpus: {len(text):,} characters, vocabulary of {len(chars)}.")
    print(f'Sample line: "{text.splitlines()[0]}"')

    # Baseline: predict the next character from the current one only (a bigram table).
    counts = torch.ones(len(chars), len(chars))  # add-one smoothing
    for a, b in zip(data[:-1], data[1:]):
        counts[a, b] += 1
    probs = counts / counts.sum(1, keepdim=True)
    bigram_nll = -torch.log(probs[val_data[:-1], val_data[1:]]).mean().item()

    results = {"bigram baseline": math.exp(bigram_nll)}
    samples = {}
    for cell in ("RNN", "LSTM", "GRU"):
        torch.manual_seed(4)
        model = CharModel(cell, len(chars))
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        crit = nn.CrossEntropyLoss()
        for _ in range(600):
            x, y = batches(train_data)
            opt.zero_grad()
            logits, _ = model(x)
            crit(logits.reshape(-1, len(chars)), y.reshape(-1)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Part 4's lesson
            opt.step()

        with torch.no_grad():
            losses = []
            for _ in range(40):
                x, y = batches(val_data)
                logits, _ = model(x)
                losses.append(crit(logits.reshape(-1, len(chars)), y.reshape(-1)).item())
            results[cell] = math.exp(float(np.mean(losses)))

            # Generate, feeding each prediction back in as the next input.
            ctx = torch.tensor([[stoi["\n"]]])
            state, generated = None, []
            for _ in range(160):
                logits, state = model(ctx, state)
                nxt = torch.multinomial(torch.softmax(logits[0, -1], dim=-1), 1)
                generated.append(itos[nxt.item()])
                ctx = nxt.view(1, 1)
            samples[cell] = "".join(generated)

    print(f"\n{'Model':<22}{'validation perplexity':>24}")
    print("-" * 74)
    for name, ppl in results.items():
        print(f"{name:<22}{ppl:>24.3f}")

    print(
        f"\nPerplexity = exp(average cross-entropy loss). Read it as 'how many characters is\n"
        f"the model effectively choosing between at each step'. A perplexity of 1.0 is\n"
        f"perfect prediction; the vocabulary size ({len(chars)}) is the score of a model that has\n"
        f"learned nothing at all. All three recurrent models land near 1.28 — on a grammar\n"
        f"this regular they are equally capable, and the architecture differences that\n"
        f"mattered in Part 3 do not show up in a task with only short-range dependencies.\n\n"
        "The bigram baseline can only see the PREVIOUS character, so at 'the quiet r...' it\n"
        "has no idea a noun is due. The recurrent models carry a hidden state and can track\n"
        "position within the sentence, which is exactly the information the grammar needs."
    )

    # Did it learn the grammar? Check generated sentences against the actual rules.
    pattern = re.compile(
        rf"^the ({'|'.join(GRAMMAR_ADJ)}) ({'|'.join(GRAMMAR_NOUN)}) "
        rf"({'|'.join(GRAMMAR_VERB)}) the ({'|'.join(GRAMMAR_ADJ)}) "
        rf"({'|'.join(GRAMMAR_NOUN)}) \.$")
    print(f"\n{'Model':<10}{'grammatical lines generated':>30}")
    print("-" * 74)
    for cell, sample in samples.items():
        lines = [l for l in sample.split("\n")[1:-1] if l]
        ok = sum(bool(pattern.match(l)) for l in lines)
        print(f"{cell:<10}{f'{ok} / {len(lines)}':>30}")

    print("\nGenerated samples (first two complete lines from each):\n")
    for cell, sample in samples.items():
        lines = [l for l in sample.split("\n")[1:-1] if l][:2]
        print(f"  {cell}:")
        for l in lines:
            print(f'    "{l}"')

    print(
        "\nNothing told these models what a word is, that spaces separate words, or that\n"
        "sentences end with a full stop. They saw a stream of characters and learned to\n"
        "predict the next one — and word boundaries, spelling and word order fell out of\n"
        "that single objective.\n\n"
        "That is exactly the objective a large language model is trained on. The\n"
        "differences are scale (billions of characters instead of 60,000) and architecture:\n"
        "project 10 replaces this recurrence with attention, which removes the sequential\n"
        "bottleneck that makes this loop impossible to parallelize."
    )

    plt.figure(figsize=(7.5, 4.4))
    names = list(results.keys())
    plt.bar(names, [results[n] for n in names],
            color=["grey", "C0", "C1", "C2"][:len(names)])
    plt.ylabel("Validation perplexity (lower is better)")
    plt.axhline(len(chars), color="red", linestyle=":", label=f"no knowledge ({len(chars)})")
    plt.axhline(1.0, color="green", linestyle=":", label="perfect (1.0)")
    plt.title("Character-level language modelling")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "language_model.png", dpi=120)
    plt.close()
    print("Saved plot to outputs/language_model.png")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    run_scratch_rnn_demo()
    run_gradient_flow_demo()
    run_memory_task_demo()
    run_exploding_gradient_demo()
    run_lstm_gates_demo()
    run_language_model_demo()
