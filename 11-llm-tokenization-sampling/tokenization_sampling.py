"""
LLM Fundamentals — how text becomes numbers (tokenization), and how a probability
distribution becomes text (sampling). Six experiments:

  Part 1  Byte-Pair Encoding trained from scratch, with an exact round-trip check
  Part 2  word vs character vs subword: the three-way tradeoff, measured
  Part 3  how GPT-2 really tokenizes things — and why LLMs miscount letters
  Part 4  temperature, top-k and top-p implemented from scratch on real logits
  Part 5  the quality/diversity tradeoff, measured on a grammar we can check
  Part 6  why greedy decoding falls into repetition loops

No API key needed, on purpose: sampling operates on the full probability vector over
the vocabulary, which a hosted API never exposes. Everything here runs locally.

Run:
    python tokenization_sampling.py
"""

import math
import os
import re
from collections import Counter
from pathlib import Path

try:  # macOS python.org builds need certifi wired up for downloads
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

OUTPUT_DIR = Path(__file__).parent / "outputs"
torch.manual_seed(0)
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Part 1 — Byte-Pair Encoding from scratch
# ---------------------------------------------------------------------------


class BPETokenizer:
    """
    Byte-Pair Encoding. Training is three lines of idea:

        1. start with every character as its own token
        2. count all adjacent PAIRS; merge the most frequent one into a new token   (1)
        3. repeat until the vocabulary is the size you want

    Frequent sequences ("ing", "the ") become single tokens; rare ones stay split
    into pieces. Nothing is hand-designed — the merges are learned from the corpus,
    which is why a tokenizer trained on English handles English efficiently and
    everything else badly (Part 3).
    """

    def __init__(self):
        self.merges: list[tuple[str, str]] = []  # in the order they were learned
        self.vocab: dict[str, int] = {}

    @staticmethod
    def _pair_counts(words: dict[tuple, int]) -> Counter:
        counts = Counter()
        for symbols, freq in words.items():
            for a, b in zip(symbols, symbols[1:]):
                counts[(a, b)] += freq
        return counts

    @staticmethod
    def _merge(words: dict[tuple, int], pair: tuple) -> dict[tuple, int]:
        merged_symbol = pair[0] + pair[1]
        out = {}
        for symbols, freq in words.items():
            new, i = [], 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                    new.append(merged_symbol)
                    i += 2
                else:
                    new.append(symbols[i])
                    i += 1
            out[tuple(new)] = out.get(tuple(new), 0) + freq
        return out

    def train(self, text: str, n_merges: int, verbose_first: int = 0):
        # "_" marks a word boundary, so the tokenizer can tell " the" from "the".
        word_freqs = Counter(text.replace("\n", " ").split())
        words = {tuple(w) + ("_",): f for w, f in word_freqs.items()}

        for step in range(n_merges):
            counts = self._pair_counts(words)
            if not counts:
                break
            best, freq = counts.most_common(1)[0]
            words = self._merge(words, best)  # (1)
            self.merges.append(best)
            if step < verbose_first:
                print(f"  merge {step + 1:>3}: {str(best):<24} "
                      f"seen {freq:>5} times  ->  '{best[0] + best[1]}'")

        # The vocabulary is the base characters plus one new symbol per merge
        # performed. (Computing it from the merged corpus instead would make it
        # appear to SHRINK as merges rise, since merging removes intermediate
        # symbols from the text without removing them from the vocabulary.)
        base = set(text.replace("\n", " ")) | {"_"}
        created = [a + b for a, b in self.merges]
        self.vocab = {s: i for i, s in enumerate(sorted(base) + created)}
        return self

    def encode(self, text: str) -> list[str]:
        """Apply the learned merges, in the order they were learned."""
        out = []
        for word in text.replace("\n", " ").split():
            symbols = list(word) + ["_"]
            for a, b in self.merges:  # order matters: earlier merges bind first
                i = 0
                while i < len(symbols) - 1:
                    if symbols[i] == a and symbols[i + 1] == b:
                        symbols[i:i + 2] = [a + b]
                    else:
                        i += 1
            out.extend(symbols)
        return out

    @staticmethod
    def decode(tokens: list[str]) -> str:
        return "".join(tokens).replace("_", " ").strip()


def _build_corpus(n_sentences=900):
    """
    Many DIFFERENT sentences from a fixed grammar, not one paragraph repeated.
    With a repetitive corpus BPE exhausts every useful merge in ~100 steps and the
    compression curve flattens for an uninteresting reason — the text simply runs
    out of distinct pairs, rather than hitting genuine diminishing returns.
    """
    rng = np.random.default_rng(11)
    adj = ["quiet", "golden", "restless", "hollow", "bright", "distant",
           "ancient", "silver", "weathered", "patient"]
    noun = ["river", "sparrow", "mountain", "lantern", "harbour", "meadow",
            "orchard", "compass", "shoreline", "thicket"]
    verb = ["carries", "remembers", "hides", "follows", "shelters", "answers",
            "guards", "mirrors", "outlasts", "gathers"]
    return "".join(
        f"the {rng.choice(adj)} {rng.choice(noun)} {rng.choice(verb)} "
        f"the {rng.choice(adj)} {rng.choice(noun)} .\n" for _ in range(n_sentences))


CORPUS = _build_corpus()


def run_bpe_demo() -> None:
    print("=" * 74)
    print("PART 1 — Byte-Pair Encoding, trained from scratch")
    print("=" * 74)

    print(f"\nTraining on {len(CORPUS):,} characters. The first merges it learns:\n")
    tok = BPETokenizer().train(CORPUS, n_merges=60, verbose_first=8)

    sample = "the quiet river carries the golden lantern ."
    encoded = tok.encode(sample)
    decoded = tok.decode(encoded)

    print(f"\nEncoding: \"{sample}\"")
    print(f"  {len(sample)} characters -> {len(encoded)} tokens")
    print(f"  tokens: {encoded[:14]}{' ...' if len(encoded) > 14 else ''}")
    print(f"\nRound trip: decode(encode(x)) == x ?  {decoded == sample}")
    print(f"  decoded: \"{decoded}\"")

    # Compression as the vocabulary grows: the central tradeoff of tokenization.
    print(f"\n{'merges':>8}{'vocab size':>13}{'tokens for corpus':>20}{'chars/token':>14}")
    print("-" * 74)
    merge_counts, ratios = [], []
    for n in (0, 10, 25, 50, 100, 200, 400):
        t = BPETokenizer().train(CORPUS, n_merges=n)
        n_tokens = len(t.encode(CORPUS))
        merge_counts.append(n)
        ratios.append(len(CORPUS) / n_tokens)
        print(f"{n:>8}{len(t.vocab):>13}{n_tokens:>20,}{len(CORPUS) / n_tokens:>14.2f}")

    print(
        "\nMore merges means a bigger vocabulary and fewer tokens per sentence. That is the\n"
        "whole tradeoff: a large vocabulary makes sequences short (cheap to process, since\n"
        "attention costs O(n^2) in sequence length — project 10) but makes the embedding\n"
        "table and output softmax large, and leaves rare tokens with too few examples to\n"
        "learn good representations for.\n\n"
        "Note the diminishing returns. Doubling the merges does not halve the token count,\n"
        "because after the common patterns are absorbed each new merge fires less often.\n"
        "Real tokenizers land around 32k-128k merges as a compromise."
    )

    plt.figure(figsize=(7.5, 4.4))
    plt.plot(merge_counts, ratios, marker="o")
    plt.xlabel("Number of BPE merges (vocabulary size)")
    plt.ylabel("Characters per token (compression)")
    plt.title("Each merge buys less than the one before it")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bpe_compression.png", dpi=120)
    plt.close()
    print("Saved plot to outputs/bpe_compression.png")


# ---------------------------------------------------------------------------
# Part 2 — word vs character vs subword
# ---------------------------------------------------------------------------


def run_granularity_demo() -> None:
    print()
    print("=" * 74)
    print("PART 2 — Word, character, or subword? The tradeoff, measured")
    print("=" * 74)

    train_text = CORPUS
    # Held-out text containing words the tokenizer has NEVER seen.
    test_text = ("the quiet river carries the unbelievable kaleidoscope . "
                 "the restless sparrow remembers the luminous cathedral .")
    unseen = ["unbelievable", "kaleidoscope", "luminous", "cathedral"]

    word_vocab = set(train_text.split())
    test_words = test_text.split()
    oov = [w for w in test_words if w not in word_vocab]

    char_vocab = set(train_text)
    bpe = BPETokenizer().train(train_text, n_merges=100)

    print(f"\nTest sentence contains {len(unseen)} words never seen in training: {unseen}\n")
    print(f"{'Tokenizer':<16}{'vocab size':>12}{'tokens':>9}{'OOV / unknown':>16}{'can represent?':>17}")
    print("-" * 74)
    print(f"{'word-level':<16}{len(word_vocab):>12}{len(test_words):>9}"
          f"{f'{len(oov)} of {len(test_words)}':>16}{'NO':>17}")
    print(f"{'character':<16}{len(char_vocab):>12}{len(test_text):>9}{'0':>16}{'yes':>17}")
    print(f"{'BPE (100 merges)':<16}{len(bpe.vocab):>12}{len(bpe.encode(test_text)):>9}"
          f"{'0':>16}{'yes':>17}")

    print(f"\nHow BPE handles a word it has never seen — it falls back to pieces:")
    for w in unseen[:2]:
        print(f"  '{w}' -> {bpe.encode(w)}")

    print(
        "\nRead the table as three ways to lose:\n"
        "  - WORD-LEVEL has short sequences but cannot represent an unseen word at all.\n"
        "    Every one becomes <UNK> and the information is gone. English has an unbounded\n"
        "    vocabulary (names, typos, compounds), so this is fatal in practice.\n"
        "  - CHARACTER-LEVEL has a tiny vocabulary and never fails, but sequences become\n"
        "    very long — and with attention costing O(n^2), 5x longer means 25x the work.\n"
        "    It also forces the model to relearn spelling from scratch.\n"
        "  - BPE (subword) gets both: a fixed vocabulary that can still spell ANY word by\n"
        "    falling back to smaller pieces, at a sequence length close to word-level.\n\n"
        "That is why every modern LLM uses a subword tokenizer. Note it is not magic — the\n"
        "unseen words above cost several tokens each, so text unlike the training corpus is\n"
        "genuinely more expensive to process. Part 3 measures how much."
    )


# ---------------------------------------------------------------------------
# Part 3 — how a real tokenizer behaves
# ---------------------------------------------------------------------------


def run_real_tokenizer_demo() -> None:
    print()
    print("=" * 74)
    print("PART 3 — How GPT-2 actually tokenizes, and what it explains")
    print("=" * 74)

    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception as exc:  # offline, or download blocked
        print(f"\n(Skipping: could not load the GPT-2 tokenizer — {type(exc).__name__}.)")
        print("This part needs a one-off download; everything else runs offline.")
        return

    print(f"\nGPT-2's real tokenizer: {enc.n_vocab:,} tokens, learned by BPE on web text.\n")

    samples = [
        ("plain English", "The quick brown fox jumps over the lazy dog."),
        ("rare/technical", "Perihelion precession anomalies in astrophysics."),
        ("a number", "The answer is 1234567890."),
        ("code", "for i in range(10): print(i**2)"),
        ("non-English", "El rápido zorro marrón salta sobre el perro."),
        ("emoji", "I love pizza 🍕🍕🍕"),
    ]
    print(f"{'Content':<18}{'chars':>7}{'tokens':>8}{'chars/token':>14}")
    print("-" * 74)
    labels, efficiency = [], []
    for label, text in samples:
        ids = enc.encode(text)
        labels.append(label)
        efficiency.append(len(text) / len(ids))
        print(f"{label:<18}{len(text):>7}{len(ids):>8}{len(text) / len(ids):>14.2f}")

    print(
        f"\nEnglish gets ~4 characters per token; other languages and emoji get far fewer.\n"
        f"This is not a neutral engineering detail — it means the SAME sentence costs several\n"
        f"times more to process in Spanish than in English, and API pricing is per token.\n"
        f"The tokenizer's training corpus was mostly English, so English is what it compresses\n"
        f"well. Tokenization is where a lot of hidden bias in LLM systems lives."
    )

    # The famous failure modes, explained by tokenization.
    print(f"\nWhy LLMs are bad at spelling and arithmetic — look at the pieces:\n")
    for word in ["strawberry", "hello", "1234567890", "3141592653"]:
        ids = enc.encode(word)
        pieces = [enc.decode([i]) for i in ids]
        print(f"  {word:<12} -> {pieces}")

    r_count = "strawberry".count("r")
    print(
        f"\n'strawberry' contains {r_count} r's, but the model never sees the letters — it sees\n"
        f"{len(enc.encode('strawberry'))} opaque token IDs. Asking it to count letters is like asking you to count\n"
        f"the strokes in a Chinese character you can only recognize as a whole. This is the\n"
        f"real explanation for the 'how many r's in strawberry' failure, and it is a\n"
        f"TOKENIZATION problem, not a reasoning one.\n\n"
        f"Numbers split just as arbitrarily, and the split depends on the digits, so '1234'\n"
        f"and '1235' may tokenize into different shapes. That is a large part of why\n"
        f"arithmetic is unreliable: the model is not given digits, it is given chunks."
    )

    # Whitespace and capitalization change the token, which surprises people.
    print(f"\nThe same word is a DIFFERENT token depending on its context:\n")
    for variant in ["hello", " hello", "Hello", " Hello", "HELLO"]:
        ids = enc.encode(variant)
        print(f"  {repr(variant):<10} -> ids {ids}")
    print(
        "\nA leading space is part of the token. This is why prompts ending in a space often\n"
        "behave worse: you have forced the model to continue with a token that never occurs\n"
        "after a space in training."
    )

    plt.figure(figsize=(7.5, 4.4))
    plt.barh(labels, efficiency, color="steelblue")
    plt.axvline(4.0, color="grey", linestyle=":", label="~4 chars/token (typical English)")
    plt.xlabel("Characters per token (higher = cheaper to process)")
    plt.title("The same tokenizer treats different content very differently")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tokenizer_efficiency.png", dpi=120)
    plt.close()
    print("Saved plot to outputs/tokenizer_efficiency.png")


# ---------------------------------------------------------------------------
# Sampling — the strategies, from scratch
# ---------------------------------------------------------------------------


def softmax_with_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """
    p_i = exp(z_i / T) / sum_j exp(z_j / T)                                     (2)

    T = 1 leaves the distribution as the model produced it. T < 1 sharpens it
    (dividing by a small number magnifies differences); T > 1 flattens it. As
    T -> 0 this becomes argmax; as T -> infinity it becomes uniform.
    """
    z = logits / max(T, 1e-8)
    z = z - z.max()  # for numerical stability, exactly as in project 06's softmax
    e = np.exp(z)
    return e / e.sum()


def top_k_filter(probs: np.ndarray, k: int) -> np.ndarray:
    """Keep the k most likely tokens, zero the rest, renormalize.              (3)"""
    if k <= 0 or k >= len(probs):
        return probs
    out = np.zeros_like(probs)
    idx = np.argpartition(probs, -k)[-k:]
    out[idx] = probs[idx]
    return out / out.sum()


def top_p_filter(probs: np.ndarray, p: float) -> np.ndarray:
    """
    Nucleus sampling: keep the smallest set of tokens whose probabilities sum
    to at least p, zero the rest, renormalize.                                 (4)

    Unlike top-k this adapts: where the model is confident the nucleus may be a
    single token; where it is unsure it may be hundreds.
    """
    order = np.argsort(-probs)
    cumulative = np.cumsum(probs[order])
    cutoff = int(np.searchsorted(cumulative, p) + 1)
    out = np.zeros_like(probs)
    keep = order[:cutoff]
    out[keep] = probs[keep]
    return out / out.sum()


def entropy_bits(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def run_sampling_math_demo() -> None:
    print()
    print("=" * 74)
    print("PART 4 — Temperature, top-k and top-p, on a real distribution")
    print("=" * 74)

    # A plausible next-token distribution: a few good options and a long bad tail.
    vocab = ["the", "a", "my", "his", "green", "quantum", "xylophone", "zzz"]
    logits = np.array([6.0, 5.2, 4.1, 3.6, 1.2, 0.4, -1.0, -3.0])

    print(f"\nModel logits for the next token:\n")
    print(f"{'token':<12}{'logit':>8}" + "".join(f"{f'T={t}':>10}" for t in (0.5, 1.0, 1.5)))
    print("-" * 74)
    dists = {T: softmax_with_temperature(logits, T) for T in (0.5, 1.0, 1.5)}
    for i, tok in enumerate(vocab):
        print(f"{tok:<12}{logits[i]:>8.1f}" + "".join(f"{dists[T][i]:>10.4f}"
                                                     for T in (0.5, 1.0, 1.5)))
    print(f"{'entropy (bits)':<12}{'':>8}" + "".join(f"{entropy_bits(dists[T]):>10.3f}"
                                                    for T in (0.5, 1.0, 1.5)))

    print(
        "\nTemperature divides the logits BEFORE the softmax. Dividing by 0.5 doubles every\n"
        "gap, so the leader pulls further ahead and entropy falls; dividing by 1.5 shrinks\n"
        "the gaps and spreads the probability out. Nothing about the model changed — this\n"
        "is a knob on the sampling, applied after the forward pass."
    )

    p1 = softmax_with_temperature(logits, 1.0)
    print(f"\nNow the two truncation methods, applied to the T=1.0 distribution:\n")
    print(f"{'method':<22}{'tokens kept':>13}{'entropy':>10}{'P(worst 3 tokens)':>20}")
    print("-" * 74)
    tail = slice(-3, None)
    for name, filtered in (
        ("none", p1),
        ("top-k, k=3", top_k_filter(p1, 3)),
        ("top-p, p=0.9", top_p_filter(p1, 0.9)),
        ("top-p, p=0.5", top_p_filter(p1, 0.5)),
    ):
        print(f"{name:<22}{int((filtered > 0).sum()):>13}{entropy_bits(filtered):>10.3f}"
              f"{filtered[tail].sum():>20.4f}")

    print(
        "\nBoth methods delete the tail, and the tail is where nonsense lives: 'xylophone'\n"
        "and 'zzz' each have small probability, but there are thousands of such tokens in a\n"
        "real vocabulary, so their COMBINED probability is not small. Sample long enough and\n"
        "you will hit one, and one bad token derails everything after it.\n\n"
        "top-k keeps a FIXED number, which is wrong in both directions: too permissive when\n"
        "the model is confident (k=50 when only 2 tokens make sense) and too restrictive\n"
        "when it is genuinely uncertain. top-p keeps a FIXED PROBABILITY MASS, so the number\n"
        "of candidates adapts to how confident the model is. That is why top-p (nucleus)\n"
        "sampling is the more common default."
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    x = np.arange(len(vocab))
    for T in (0.5, 1.0, 1.5):
        ax1.plot(x, dists[T], marker="o", label=f"T = {T}")
    ax1.set_xticks(x); ax1.set_xticklabels(vocab, rotation=45, fontsize=7)
    ax1.set_ylabel("probability"); ax1.set_title("Temperature reshapes the distribution", fontsize=10)
    ax1.legend(fontsize=8)
    temps = np.linspace(0.1, 3.0, 40)
    ax2.plot(temps, [entropy_bits(softmax_with_temperature(logits, t)) for t in temps])
    ax2.axhline(math.log2(len(vocab)), color="grey", linestyle=":",
                label=f"uniform ({math.log2(len(vocab)):.2f} bits)")
    ax2.set_xlabel("temperature"); ax2.set_ylabel("entropy (bits)")
    ax2.set_title("T -> 0 is argmax; T -> inf is uniform", fontsize=10)
    ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sampling_distributions.png", dpi=120)
    plt.close()
    print("Saved plot to outputs/sampling_distributions.png")


# ---------------------------------------------------------------------------
# Parts 5 & 6 — sampling a real model
# ---------------------------------------------------------------------------

ADJ = ["quiet", "golden", "restless", "hollow", "bright", "distant"]
NOUN = ["river", "sparrow", "mountain", "lantern", "harbour", "meadow"]
VERB = ["carries", "remembers", "hides", "follows", "shelters", "answers"]
GRAMMAR = re.compile(rf"^the ({'|'.join(ADJ)}) ({'|'.join(NOUN)}) ({'|'.join(VERB)}) "
                     rf"the ({'|'.join(ADJ)}) ({'|'.join(NOUN)}) \.$")


def make_corpus(n=1500):
    rng = np.random.default_rng(7)
    return "".join(f"the {rng.choice(ADJ)} {rng.choice(NOUN)} {rng.choice(VERB)} "
                   f"the {rng.choice(ADJ)} {rng.choice(NOUN)} .\n" for _ in range(n))


# The context length. Training and the positional-embedding table MUST use the same
# value: with learned positions, any index never seen in training holds a random
# vector, and generation degenerates into noise the moment it reaches one. (This bug
# was in an earlier version of this file — max_len was 64 while training used 48, so
# the first generated line was perfect and everything after it was garbage. It is
# also exactly why project 10's SINUSOIDAL encoding is preferred: being a fixed
# function, it extrapolates to positions never trained on.)
SEQ_LEN = 64


class MiniGPT(nn.Module):
    """Project 10's transformer, unchanged in structure — now used as a text source."""

    def __init__(self, vocab, d_model=64, n_heads=4, n_layers=2, max_len=SEQ_LEN):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                           batch_first=True, dropout=0.0, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, vocab)
        self.max_len = max_len

    def forward(self, x):
        n = x.shape[1]
        h = self.embed(x) + self.pos(torch.arange(n, device=x.device))
        mask = nn.Transformer.generate_square_subsequent_mask(n, device=x.device)
        return self.head(self.blocks(h, mask=mask, is_causal=True))


def train_mini_gpt():
    text = make_corpus()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    model = MiniGPT(len(chars))
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    crit = nn.CrossEntropyLoss()
    seq_len = SEQ_LEN
    for _ in range(2500):
        ix = torch.randint(0, len(data) - seq_len - 1, (64,))
        x = torch.stack([data[i:i + seq_len] for i in ix])
        y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix])
        opt.zero_grad()
        crit(model(x).reshape(-1, len(chars)), y.reshape(-1)).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model, stoi, {i: c for c, i in stoi.items()}


@torch.no_grad()
def generate(model, stoi, itos, n_chars=600, T=1.0, k=0, p=0.0, greedy=False):
    ctx = torch.tensor([[stoi["\n"]]])
    out = []
    for _ in range(n_chars):
        logits = model(ctx[:, -model.max_len:])[0, -1].numpy()
        if greedy:
            nxt = int(np.argmax(logits))
        else:
            probs = softmax_with_temperature(logits, T)
            if k:
                probs = top_k_filter(probs, k)
            if p:
                probs = top_p_filter(probs, p)
            nxt = int(RNG.choice(len(probs), p=probs))
        out.append(itos[nxt])
        ctx = torch.cat([ctx, torch.tensor([[nxt]])], dim=1)
    return "".join(out)


def score(text: str):
    """Grammaticality and diversity — the two things sampling trades against."""
    lines = [l for l in text.split("\n")[1:-1] if l.strip()]
    if not lines:
        return 0.0, 0.0, 0
    valid = sum(bool(GRAMMAR.match(l)) for l in lines)
    return valid / len(lines), len(set(lines)) / len(lines), len(lines)


def run_sampling_quality_demo(model, stoi, itos) -> None:
    print()
    print("=" * 74)
    print("PART 5 — The quality/diversity tradeoff, on a grammar we can check")
    print("=" * 74)

    print("\nThe model was trained on sentences of the form")
    print("  'the ADJ NOUN VERB the ADJ NOUN .'")
    print("so a regex can tell whether each generated line is actually grammatical —")
    print("no human judgement needed. Diversity is the fraction of lines that are unique.\n")
    print(f"{'strategy':<24}{'lines':>7}{'grammatical':>14}{'unique':>10}")
    print("-" * 74)

    configs = [
        ("greedy (argmax)", dict(greedy=True)),
        ("T = 0.3", dict(T=0.3)),
        ("T = 0.7", dict(T=0.7)),
        ("T = 1.0 (raw model)", dict(T=1.0)),
        ("T = 1.5", dict(T=1.5)),
        ("T = 2.0", dict(T=2.0)),
        ("T = 1.0, top-k = 5", dict(T=1.0, k=5)),
        ("T = 1.0, top-p = 0.9", dict(T=1.0, p=0.9)),
        ("T = 1.5, top-p = 0.9", dict(T=1.5, p=0.9)),
    ]
    results = {}
    for name, kwargs in configs:
        text = generate(model, stoi, itos, n_chars=2500, **kwargs)
        g, d, n = score(text)
        results[name] = (g, d)
        print(f"{name:<24}{n:>7}{g:>14.3f}{d:>10.3f}")

    print(
        "\nRead the two columns against each other — that IS the tradeoff:\n"
        "  - GREEDY is the extreme case: perfect grammar, and almost no unique lines. It\n"
        "    always takes the argmax, so it is the safest and the most repetitive.\n"
        "  - TEMPERATURE degrades grammar monotonically: 1.00 at T=0.3, 0.80 at T=1.0,\n"
        "    0.09 at T=2.0. Flattening the distribution hands probability to tokens the\n"
        "    model rated unlikely, and on this task 'unlikely' means 'ungrammatical'.\n"
        "  - Note that diversity stays HIGH even at low temperature here (0.96 at T=0.3).\n"
        "    That is a property of this grammar, which offers many equally good\n"
        "    continuations, not a general rule — on text with one obvious continuation,\n"
        "    low temperature collapses diversity the way greedy does.\n"
        "  - TOP-P is the row that matters: at T=1.0 it lifts grammar from 0.80 to 0.94\n"
        "    while keeping diversity at 1.00. It removes the bad tail without removing the\n"
        "    genuine choice, which is why real systems combine temperature with nucleus\n"
        "    sampling rather than picking one.\n"
        "  - TOP-K does worse than top-p here (0.72 vs 0.94) — with a character-level\n"
        "    vocabulary, a fixed k=5 is far too permissive at the many positions where only\n"
        "    one or two characters are valid. This is the adaptivity argument, measured.\n\n"
        "There is no universally correct setting. Code generation and factual answers want\n"
        "low temperature; brainstorming and fiction want high. The parameter is a statement\n"
        "about what kind of mistake you can tolerate."
    )

    plt.figure(figsize=(7.5, 5))
    for name, (g, d) in results.items():
        plt.scatter(d, g, s=60)
        plt.annotate(name, (d, g), fontsize=7, xytext=(4, 4), textcoords="offset points")
    plt.xlabel("Diversity (fraction of generated lines that are unique)")
    plt.ylabel("Quality (fraction that are grammatical)")
    plt.title("Every sampling setting is a point on this tradeoff")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "quality_diversity.png", dpi=120)
    plt.close()
    print("Saved plot to outputs/quality_diversity.png")


def run_repetition_demo(model, stoi, itos) -> None:
    print()
    print("=" * 74)
    print("PART 6 — Why greedy decoding gets stuck")
    print("=" * 74)

    greedy_text = generate(model, stoi, itos, n_chars=400, greedy=True)
    sampled_text = generate(model, stoi, itos, n_chars=400, T=1.0, p=0.9)

    def repetition_rate(text):
        lines = [l for l in text.split("\n") if l.strip()]
        return 1 - len(set(lines)) / max(len(lines), 1)

    print(f"\n{'strategy':<24}{'repeated lines':>18}")
    print("-" * 74)
    print(f"{'greedy':<24}{repetition_rate(greedy_text):>18.3f}")
    print(f"{'top-p = 0.9':<24}{repetition_rate(sampled_text):>18.3f}")

    print(f"\nGreedy output (first 3 lines):")
    for l in [l for l in greedy_text.split("\n") if l.strip()][:3]:
        print(f'    "{l}"')
    print(f"\nTop-p output (first 3 lines):")
    for l in [l for l in sampled_text.split("\n") if l.strip()][:3]:
        print(f'    "{l}"')

    print(
        "\nGreedy decoding is DETERMINISTIC: given the same context it always picks the same\n"
        "token. So once it produces a line, the context that follows resembles the context\n"
        "before it, and it produces the same line again — a fixed point it cannot escape,\n"
        "because escaping would require choosing a token that is not the argmax.\n\n"
        "This is not a flaw in this small model; it is why production systems never use\n"
        "pure greedy decoding for open-ended text. Note the tension with Part 5, though:\n"
        "greedy scored well on grammar precisely BECAUSE it always picks the safest token.\n"
        "High quality and zero diversity are the same behaviour measured two ways."
    )


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    run_bpe_demo()
    run_granularity_demo()
    run_real_tokenizer_demo()
    run_sampling_math_demo()
    print("\n(training a small GPT on the grammar corpus for Parts 5 and 6...)")
    model, stoi, itos = train_mini_gpt()
    run_sampling_quality_demo(model, stoi, itos)
    run_repetition_demo(model, stoi, itos)
