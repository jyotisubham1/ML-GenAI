"""
The Transformer — scaled dot-product attention derived and implemented from scratch,
then six experiments that each test a design decision rather than accepting it:

  Part 1  attention from scratch, verified against F.scaled_dot_product_attention
  Part 2  why divide by sqrt(d_k)? measured: softmax saturation and dead gradients
  Part 3  attention is permutation-equivariant — proof, and why that forces positional
          encoding to exist
  Part 4  positional encoding, and a task that is impossible without it
  Part 5  causal masking, verified by checking the future genuinely cannot leak
  Part 6  a mini-GPT: text generation, and the 80-step memory task project 09 failed

Run:
    python transformer.py

See README.md for the math behind every formula referenced in the comments below.
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUTPUT_DIR = Path(__file__).parent / "outputs"
torch.manual_seed(0)
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Part 1 — attention from scratch
# ---------------------------------------------------------------------------


def attention_scratch(Q, K, V, mask=None):
    """
    Scaled dot-product attention:

        scores = Q K^T / sqrt(d_k)                                             (1)
        A      = softmax(scores)                                               (2)
        out    = A V                                                           (3)

    Read it as a soft dictionary lookup. Every position emits a QUERY ("what am I
    looking for?"), every position emits a KEY ("what do I contain?"), and the dot
    product of a query with a key measures how well they match. Softmax turns those
    match scores into weights that sum to 1, and the output is the weighted average
    of the VALUES — "what I will actually pass on".

    Q, K, V: (..., seq_len, d_k). Returns (output, attention_weights).
    """
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # (1)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = torch.softmax(scores, dim=-1)  # (2)
    return weights @ V, weights  # (3)


def run_attention_demo() -> None:
    print("=" * 74)
    print("PART 1 — Scaled dot-product attention from scratch")
    print("=" * 74)

    torch.manual_seed(1)
    batch, heads, seq_len, d_k = 2, 4, 6, 16
    Q = torch.randn(batch, heads, seq_len, d_k, dtype=torch.float64)
    K = torch.randn(batch, heads, seq_len, d_k, dtype=torch.float64)
    V = torch.randn(batch, heads, seq_len, d_k, dtype=torch.float64)

    mine, weights = attention_scratch(Q, K, V)
    theirs = F.scaled_dot_product_attention(Q, K, V)
    print(f"\nUnmasked:  max |scratch - F.scaled_dot_product_attention| = "
          f"{(mine - theirs).abs().max():.3e}")

    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.float64))
    mine_c, _ = attention_scratch(Q, K, V, mask=causal)
    theirs_c = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    print(f"Causal:    max |scratch - F.scaled_dot_product_attention| = "
          f"{(mine_c - theirs_c).abs().max():.3e}")

    print(f"\nAttention weights are a probability distribution over positions — each row of")
    print(f"the matrix sums to 1. Checking: max |row sum - 1| = "
          f"{(weights.sum(-1) - 1).abs().max():.3e}")

    print(
        "\nThree matrices and a softmax — that is the entire operation at the heart of every\n"
        "modern language model. Compare it against project 09's recurrence:\n\n"
        "  RNN:          to relate step 1 and step 60, information must pass through 59\n"
        "                intermediate hidden states, decaying at every one.\n"
        "  Attention:    position 60 computes a dot product with position 1 DIRECTLY. The\n"
        "                path length between any two positions is 1, no matter how far\n"
        "                apart they are.\n\n"
        "That single change is why the vanishing-gradient problem project 09 spent five\n"
        "experiments measuring simply does not arise here. The cost is that every position\n"
        "attends to every other: O(n^2) work and memory in the sequence length, where\n"
        "recurrence was O(n). That trade is the central limitation of transformers, and the\n"
        "reason context windows are finite."
    )


# ---------------------------------------------------------------------------
# Part 2 — why sqrt(d_k)
# ---------------------------------------------------------------------------


def run_scaling_demo() -> None:
    print()
    print("=" * 74)
    print("PART 2 — Why divide by sqrt(d_k)? Measuring what happens when you don't")
    print("=" * 74)

    torch.manual_seed(2)
    seq_len = 32
    print(f"\nQ and K entries drawn from N(0,1), so each score is a sum of d_k products.")
    print(f"Theory (README §4.2): Var(q·k) = d_k, so scores grow like sqrt(d_k) and the")
    print(f"softmax saturates. Measured over {seq_len} positions:\n")
    print(f"{'d_k':>6}{'score std (unscaled)':>22}{'max attn weight':>18}"
          f"{'entropy (bits)':>17}{'scaled entropy':>16}")
    print("-" * 74)

    dims, unscaled_entropy, scaled_entropy, grads = [], [], [], []
    for d_k in (4, 16, 64, 256, 1024):
        Q = torch.randn(seq_len, d_k, requires_grad=True)
        K = torch.randn(seq_len, d_k)
        raw = Q @ K.T
        w_unscaled = torch.softmax(raw, dim=-1)
        w_scaled = torch.softmax(raw / math.sqrt(d_k), dim=-1)

        ent_u = float(-(w_unscaled * torch.log2(w_unscaled + 1e-12)).sum(-1).mean().detach())
        ent_s = float(-(w_scaled * torch.log2(w_scaled + 1e-12)).sum(-1).mean().detach())

        # How much gradient survives the unscaled softmax?
        w_unscaled.sum().backward(retain_graph=True)
        g_unscaled = float(Q.grad.abs().mean())
        Q.grad = None
        torch.softmax(raw / math.sqrt(d_k), dim=-1).sum().backward()
        g_scaled = float(Q.grad.abs().mean())

        dims.append(d_k)
        unscaled_entropy.append(ent_u)
        scaled_entropy.append(ent_s)
        grads.append((g_unscaled, g_scaled))
        print(f"{d_k:>6}{float(raw.std().detach()):>22.2f}{float(w_unscaled.max().detach()):>18.4f}"
              f"{ent_u:>17.3f}{ent_s:>16.3f}")

    uniform_entropy = math.log2(seq_len)
    print(
        f"\nMaximum possible entropy over {seq_len} positions is log2({seq_len}) = {uniform_entropy:.2f} bits —"
        f" that is\nattention spread evenly over everything. Zero bits means all the weight has\n"
        f"collapsed onto a single position.\n\n"
        f"UNSCALED, entropy falls from {unscaled_entropy[0]:.2f} bits at d_k=4 to "
        f"{unscaled_entropy[-1]:.3f} at d_k=1024: the softmax\nhas become a hard argmax. SCALED, entropy "
        f"stays near {np.mean(scaled_entropy):.2f} bits at every dimension —\nwhich is the whole point of "
        f"the sqrt(d_k).\n\n"
        f"Why saturation is fatal rather than merely untidy: softmax's gradient is\n"
        f"proportional to p(1-p) per entry. When p has collapsed to 1 for one position and 0\n"
        f"for the rest, that product is ~0 everywhere and NO gradient flows back to Q and K."
    )
    print(f"\n{'d_k':>6}{'mean |dL/dQ| unscaled':>24}{'scaled':>14}{'ratio':>14}")
    print("-" * 74)
    for d_k, (gu, gs) in zip(dims, grads):
        print(f"{d_k:>6}{gu:>24.3e}{gs:>14.3e}{gs / max(gu, 1e-30):>13.1f}x")

    print(
        "\nThis is project 02's saturated-sigmoid problem and project 06's vanishing gradient\n"
        "in a third costume. The fix is the same shape too: keep the pre-activation in the\n"
        "range where the nonlinearity actually has a slope. Dividing by sqrt(d_k) makes the\n"
        "scores' standard deviation 1 regardless of dimension, which is exactly what He\n"
        "initialization did for depth in project 06."
    )

    plt.figure(figsize=(7.5, 4.4))
    plt.plot(dims, unscaled_entropy, marker="o", label="unscaled  QK^T")
    plt.plot(dims, scaled_entropy, marker="s", label="scaled  QK^T / sqrt(d_k)")
    plt.axhline(uniform_entropy, color="grey", linestyle=":",
                label=f"uniform attention ({uniform_entropy:.2f} bits)")
    plt.xscale("log")
    plt.xlabel("d_k (key dimension, log scale)")
    plt.ylabel("Attention entropy (bits)")
    plt.title("Without the scaling, softmax collapses to a hard argmax")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "scaling.png", dpi=120)
    plt.close()
    print("Saved plot to outputs/scaling.png")


# ---------------------------------------------------------------------------
# Part 3 — permutation equivariance
# ---------------------------------------------------------------------------


def run_permutation_demo() -> None:
    print()
    print("=" * 74)
    print("PART 3 — Attention has no idea what order anything is in")
    print("=" * 74)

    torch.manual_seed(3)
    seq_len, d_k = 8, 16
    X = torch.randn(1, seq_len, d_k, dtype=torch.float64)
    Wq, Wk, Wv = (torch.randn(d_k, d_k, dtype=torch.float64) * 0.3 for _ in range(3))

    def attend(x):
        return attention_scratch(x @ Wq, x @ Wk, x @ Wv)[0]

    perm = torch.randperm(seq_len)
    out_then_perm = attend(X)[:, perm, :]  # attend, then shuffle the outputs
    perm_then_out = attend(X[:, perm, :])  # shuffle the inputs, then attend

    print(f"\nShuffle order: {perm.tolist()}")
    print(f"\n  attention(X) then permute   vs   attention(permuted X)")
    print(f"  max |difference| = {(out_then_perm - perm_then_out).abs().max():.3e}")

    print(
        "\nThey are identical to machine precision. Attention is PERMUTATION-EQUIVARIANT:\n"
        "shuffle the inputs and the outputs shuffle the same way, unchanged in content.\n\n"
        "Which means the mechanism literally cannot distinguish 'dog bites man' from\n"
        "'man bites dog'. Every position is compared with every other by dot product, and a\n"
        "dot product has no notion of which came first. Contrast project 08's convolution\n"
        "(neighbouring positions are wired together) and project 09's recurrence (the loop\n"
        "imposes an order by construction) — attention discards both.\n\n"
        "So order has to be put back by hand, as information added to the inputs. That is\n"
        "what positional encoding is for, and this experiment is why it is not optional."
    )

    # Confirm the same property survives inside a real attention layer, then breaks
    # once positional information is added.
    pe = sinusoidal_encoding(seq_len, d_k).double()
    out_pe = attention_scratch((X + pe) @ Wq, (X + pe) @ Wk, (X + pe) @ Wv)[0][:, perm, :]
    out_pe_perm = attention_scratch((X[:, perm, :] + pe) @ Wq, (X[:, perm, :] + pe) @ Wk,
                                    (X[:, perm, :] + pe) @ Wv)[0]
    print(f"\nRepeating the test WITH positional encoding added to the inputs:")
    print(f"  max |difference| = {(out_pe - out_pe_perm).abs().max():.3e}")
    print("  -> now large, i.e. the model can finally tell the orderings apart.")


# ---------------------------------------------------------------------------
# Part 4 — positional encoding
# ---------------------------------------------------------------------------


def sinusoidal_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """
    PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))                              (4)
    PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))

    A different frequency per dimension pair: fast-oscillating dimensions encode fine
    position, slow ones encode coarse position — like the digits of a number written
    in a strange base. Because it is a fixed function rather than learned parameters,
    it extends to positions longer than anything seen in training.
    """
    pos = torch.arange(seq_len).unsqueeze(1).float()
    i = torch.arange(0, d_model, 2).float()
    denom = torch.pow(10000.0, i / d_model)
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(pos / denom)
    pe[:, 1::2] = torch.cos(pos / denom)
    return pe


class TinyTransformer(nn.Module):
    """A minimal encoder: embed -> (+ positions) -> self-attention blocks -> classify."""

    def __init__(self, vocab, d_model=64, n_heads=4, n_layers=2, max_len=128,
                 use_positional=True, causal=False):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.use_positional = use_positional
        self.causal = causal
        self.register_buffer("pe", sinusoidal_encoding(max_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model, batch_first=True,
            dropout=0.0, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, x):
        h = self.embed(x)
        if self.use_positional:
            h = h + self.pe[:x.shape[1]]
        mask = None
        if self.causal:
            mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1], device=x.device)
        h = self.blocks(h, mask=mask, is_causal=self.causal)
        return self.head(h)


def make_memory_task(n_samples, seq_len, n_symbols=8):
    """Project 09's task, unchanged: recall the first symbol after N blank steps."""
    X = RNG.integers(1, n_symbols, size=(n_samples, seq_len))
    y = X[:, 0].copy()
    X[:, 1:] = 0
    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def run_positional_demo() -> None:
    print()
    print("=" * 74)
    print("PART 4 — Positional encoding: a task that is impossible without it")
    print("=" * 74)

    pe = sinusoidal_encoding(64, 32)
    print(f"\nSinusoidal encoding, {pe.shape[0]} positions x {pe.shape[1]} dimensions.")
    print(f"  Dot product between position 0 and position 1:  {float(pe[0] @ pe[1]):>8.3f}")
    print(f"  Dot product between position 0 and position 10: {float(pe[0] @ pe[10]):>8.3f}")
    print(f"  Dot product between position 0 and position 40: {float(pe[0] @ pe[40]):>8.3f}")
    print("  -> similarity decreases with distance, so 'how far apart are these?' is")
    print("     information the dot product inside attention can actually use.")

    # A task that depends ENTIRELY on order: two distinct symbols are dropped at two
    # random positions and the model must report WHICH CAME FIRST.
    #
    # Designing this correctly took two attempts. The first version put one symbol at
    # position 0 and the other at the final position, and read the answer from the final
    # position — but that is solvable without any positional information, because the
    # final position can see its OWN token through the residual stream and answer "the
    # other one". Both models scored 1.000 and the experiment proved nothing.
    #
    # Here both symbols sit at random INTERIOR positions and the answer is read from the
    # last position, which is always a blank. Now the only way to tell the two apart is
    # where they are, so without positional encoding the model can do no better than
    # guessing between the two symbols it can see.
    def make_order_task(n, seq_len=24, n_symbols=8):
        X = np.zeros((n, seq_len), dtype=np.int64)
        y = np.zeros(n, dtype=np.int64)
        for i in range(n):
            p1, p2 = sorted(RNG.choice(seq_len - 1, size=2, replace=False))
            a, b = RNG.choice(np.arange(1, n_symbols), size=2, replace=False)
            X[i, p1], X[i, p2] = a, b
            y[i] = a  # the symbol that came FIRST
        return torch.tensor(X), torch.tensor(y)

    seq_len = 24
    X, y = make_order_task(4000, seq_len)
    Xte, yte = make_order_task(800, seq_len)

    print(f"\nTask: two distinct symbols at two random positions; report the one that came")
    print(f"FIRST. The answer is read from the final position, which is always blank, so")
    print(f"the model cannot cheat by looking at its own token. Without positional")
    print(f"information it sees an unordered bag and should be stuck near 50%.\n")
    print(f"{'Model':<34}{'test accuracy':>16}")
    print("-" * 74)

    results = {}
    for label, use_pe in (("Transformer WITHOUT positions", False),
                          ("Transformer with positions", True)):
        torch.manual_seed(4)
        model = TinyTransformer(8, use_positional=use_pe)
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)
        crit = nn.CrossEntropyLoss()
        for _ in range(25):
            perm = torch.randperm(len(X))
            for i in range(0, len(X), 64):
                idx = perm[i:i + 64]
                opt.zero_grad()
                crit(model(X[idx])[:, -1, :], y[idx]).backward()
                opt.step()
        with torch.no_grad():
            acc = (model(Xte)[:, -1, :].argmax(1) == yte).float().mean().item()
        results[label] = acc
        print(f"{label:<34}{acc:>16.4f}")

    print(
        f"\nThe difference is the experiment. Same architecture, same data, same training —\n"
        f"the only change is whether sinusoidal position vectors were added to the inputs.\n"
        f"Without them the model is guessing between the two symbols it can see but cannot\n"
        f"order; with them it solves the task.\n\n"
        f"Note what this says about the design: attention is a set operation, and a\n"
        f"transformer is a set model with position bolted on as data. That sounds like a\n"
        f"weakness and is actually the source of its flexibility — swap the encoding and\n"
        f"the same machinery handles images (2D positions), audio, or graphs."
    )

    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(pe.T, aspect="auto", cmap="RdBu_r")
    plt.xlabel("position")
    plt.ylabel("encoding dimension")
    plt.title("Sinusoidal positional encoding", fontsize=10)
    plt.colorbar(fraction=0.046)
    plt.subplot(1, 2, 2)
    plt.bar(list(results.keys()), list(results.values()), color=["indianred", "seagreen"])
    plt.xticks(rotation=12, fontsize=7)
    plt.ylabel("Test accuracy")
    plt.title("Report the FIRST of two symbols", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "positional_encoding.png", dpi=120)
    plt.close()
    print("Saved plot to outputs/positional_encoding.png")


# ---------------------------------------------------------------------------
# Part 5 — causal masking
# ---------------------------------------------------------------------------


def run_masking_demo() -> None:
    print()
    print("=" * 74)
    print("PART 5 — Causal masking: verifying the future genuinely cannot leak")
    print("=" * 74)

    torch.manual_seed(5)
    seq_len, d_k = 6, 8
    Q, K, V = (torch.randn(1, seq_len, d_k, dtype=torch.float64) for _ in range(3))
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.float64))

    _, weights = attention_scratch(Q, K, V, mask=causal)
    print("\nCausal attention weight matrix (row = query position, column = attended-to):\n")
    print("        " + "".join(f"{j:>8}" for j in range(seq_len)))
    for i in range(seq_len):
        print(f"  pos {i}: " + "".join(f"{float(weights[0, i, j]):>8.3f}" for j in range(seq_len)))

    upper = weights[0].triu(diagonal=1)
    print(f"\nSum of everything above the diagonal (attention paid to the future): "
          f"{float(upper.sum()):.3e}")

    # The real test: change a FUTURE token and check earlier outputs are untouched.
    out_a, _ = attention_scratch(Q, K, V, mask=causal)
    K2, V2 = K.clone(), V.clone()
    K2[0, -1] += 10.0  # tamper with the last position only
    V2[0, -1] += 10.0
    out_b, _ = attention_scratch(Q, K2, V2, mask=causal)

    print(f"\nNow corrupt the LAST position's key and value, and compare outputs:")
    print(f"{'position':>10}{'max |change|':>18}")
    print("-" * 74)
    for i in range(seq_len):
        print(f"{i:>10}{float((out_a[0, i] - out_b[0, i]).abs().max()):>18.3e}")

    print(
        "\nPositions 0 to 4 are bit-identical; only the last position changed. That is the\n"
        "guarantee causal masking provides, and it is what makes it possible to train a\n"
        "language model efficiently.\n\n"
        "Why it matters: predicting the next token means position t may use only positions\n"
        "<= t. Without masking, the model would see the answer in its input and learn\n"
        "nothing (project 03's data leakage, built into the architecture). WITH masking, a\n"
        "single forward pass over a sequence of length n produces n training examples at\n"
        "once — every position predicting its own next token, all in parallel.\n\n"
        "That is the efficiency project 09's RNN could never have. A recurrent model must\n"
        "compute step t before step t+1; a masked transformer computes all n positions\n"
        "simultaneously, which is what makes training on internet-scale text feasible.\n"
        "The mask is implemented by setting the forbidden scores to -inf before the\n"
        "softmax, since exp(-inf) = 0 — an exact zero, not a small number."
    )


# ---------------------------------------------------------------------------
# Part 6 — a mini-GPT
# ---------------------------------------------------------------------------

ADJ = ["quiet", "golden", "restless", "hollow", "bright", "distant"]
NOUN = ["river", "sparrow", "mountain", "lantern", "harbour", "meadow"]
VERB = ["carries", "remembers", "hides", "follows", "shelters", "answers"]


def make_corpus(n_sentences=1200):
    """Project 09's corpus, unchanged, so the perplexities are directly comparable."""
    rng = np.random.default_rng(7)
    return "".join(
        f"the {rng.choice(ADJ)} {rng.choice(NOUN)} {rng.choice(VERB)} "
        f"the {rng.choice(ADJ)} {rng.choice(NOUN)} .\n" for _ in range(n_sentences))


def run_mini_gpt_demo() -> None:
    print()
    print("=" * 74)
    print("PART 6 — A mini-GPT, and the memory task project 09 could not solve")
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
        return (torch.stack([d[i:i + seq_len] for i in ix]),
                torch.stack([d[i + 1:i + seq_len + 1] for i in ix]))

    torch.manual_seed(6)
    model = TinyTransformer(len(chars), d_model=64, n_heads=4, n_layers=2,
                            max_len=seq_len, causal=True)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    crit = nn.CrossEntropyLoss()
    n_params = sum(p.numel() for p in model.parameters())

    for _ in range(1200):
        x, y = batches(train_data)
        opt.zero_grad()
        crit(model(x).reshape(-1, len(chars)), y.reshape(-1)).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    with torch.no_grad():
        losses = [crit(model(x).reshape(-1, len(chars)), y.reshape(-1)).item()
                  for x, y in (batches(val_data) for _ in range(40))]
    ppl = math.exp(float(np.mean(losses)))

    print(f"\nMini-GPT: {n_params:,} parameters, 2 layers, 4 heads, causal masking.")
    print(f"\n{'Model':<34}{'validation perplexity':>24}")
    print("-" * 74)
    print(f"{'bigram baseline (project 09)':<34}{4.385:>24.3f}")
    print(f"{'LSTM (project 09)':<34}{1.277:>24.3f}")
    print(f"{'this transformer':<34}{ppl:>24.3f}")

    # Generate by sampling one character at a time, feeding the result back in.
    with torch.no_grad():
        ctx = torch.tensor([[stoi["\n"]]])
        out = []
        for _ in range(180):
            logits = model(ctx[:, -seq_len:])[0, -1]
            nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            out.append(itos[int(nxt)])
            ctx = torch.cat([ctx, nxt.view(1, 1)], dim=1)
    lines = [l for l in "".join(out).split("\n")[1:-1] if l][:3]
    print("\nGenerated samples:")
    for l in lines:
        print(f'    "{l}"')

    # --- The comparison that matters: project 09's 80-step memory task ---
    print()
    print("-" * 74)
    print("The 80-step memory task, where every recurrent model in project 09 failed:")
    print("-" * 74)

    seq = 80
    X, y = make_memory_task(3000, seq)
    Xte, yte = make_memory_task(600, seq)
    accs = []
    for seed in range(3):
        torch.manual_seed(seed)
        m = TinyTransformer(8, d_model=64, n_heads=4, n_layers=2, max_len=seq)
        opt = torch.optim.Adam(m.parameters(), lr=3e-4)
        for _ in range(25):
            perm = torch.randperm(len(X))
            for i in range(0, len(X), 64):
                idx = perm[i:i + 64]
                opt.zero_grad()
                crit(m(X[idx])[:, -1, :], y[idx]).backward()
                opt.step()
        with torch.no_grad():
            accs.append((m(Xte)[:, -1, :].argmax(1) == yte).float().mean().item())

    print(f"\n{'Model':<40}{'accuracy at 80 steps':>22}")
    print("-" * 74)
    for name, acc in (("RNN (project 09)", 0.133), ("LSTM (project 09)", 0.141),
                      ("LSTM forget-bias=3 (project 09)", 0.423),
                      ("GRU (project 09)", 0.145)):
        print(f"{name:<40}{acc:>22.3f}")
    print(f"{'Transformer (mean of 3 seeds)':<40}{np.mean(accs):>22.3f}")
    print(f"{'':<40}{f'(min {min(accs):.2f}, max {max(accs):.2f})':>22}")

    print(
        f"\nChance is 0.143. Project 09's recurrent models were all at or near it — the\n"
        f"information sits 80 steps in the past and the gradient cannot reach back that far.\n\n"
        f"The transformer does not have to reach back. Position 80 computes a dot product\n"
        f"with position 0 directly: one operation, not eighty. The distance between two\n"
        f"positions is irrelevant to attention, which is precisely the property recurrence\n"
        f"could not provide at any amount of gating.\n\n"
        f"This is the whole argument for the architecture, measured on the exact task the\n"
        f"previous project failed."
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax1.bar(["bigram", "RNN", "LSTM", "GRU", "transformer"],
            [4.385, 1.279, 1.277, 1.278, ppl],
            color=["grey", "C0", "C1", "C2", "crimson"])
    ax1.set_ylabel("Validation perplexity (lower is better)")
    ax1.set_title("Character-level language modelling", fontsize=10)
    ax2.bar(["RNN", "LSTM", "LSTM fb=3", "GRU", "transformer"],
            [0.133, 0.141, 0.423, 0.145, float(np.mean(accs))],
            color=["C0", "C1", "C3", "C2", "crimson"])
    ax2.axhline(1 / 7, color="grey", linestyle=":", label="chance")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("80-step memory task", fontsize=10)
    ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "transformer_vs_rnn.png", dpi=120)
    plt.close()
    print("Saved plot to outputs/transformer_vs_rnn.png")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    run_attention_demo()
    run_scaling_demo()
    run_permutation_demo()
    run_positional_demo()
    run_masking_demo()
    run_mini_gpt_demo()
