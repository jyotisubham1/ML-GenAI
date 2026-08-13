"""
Embeddings & Vector Search — turning text into vectors where distance means meaning,
then finding the nearest ones quickly. Six experiments:

  Part 1  cosine similarity from scratch, and why it beats euclidean for text
  Part 2  do embeddings actually capture meaning? measured against paraphrases
  Part 3  semantic vs keyword search — including where keyword search WINS
  Part 4  the curse of dimensionality, measured
  Part 5  brute force vs an approximate index: the speed/recall tradeoff
  Part 6  chunking — the boring decision that decides whether RAG works

No API key needed: embeddings run locally via sentence-transformers (Groq has no
embeddings endpoint). Downloads a ~90 MB model on first run.

Run:
    python embeddings_search.py
"""

import math
import os
import re
import time
from pathlib import Path

try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).parent / "outputs"
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Part 1 — similarity from scratch
# ---------------------------------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    cos(a, b) = (a . b) / (||a|| ||b||)                                        (1)

    The dot product measures how much two vectors point the same way; dividing by
    both lengths removes magnitude entirely, leaving only DIRECTION. Result is in
    [-1, 1]: 1 = same direction, 0 = perpendicular (unrelated), -1 = opposite.
    """
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """||a - b||_2 — straight-line distance. Sensitive to magnitude, unlike (1)."""
    return float(np.linalg.norm(a - b))


def run_similarity_demo(model) -> None:
    print("=" * 74)
    print("PART 1 — Cosine similarity, and why not euclidean distance")
    print("=" * 74)

    short = "The cat sat on the mat."
    long_same = " ".join([short] * 8)      # same MEANING, 8x the length
    different = "Quantum chromodynamics describes the strong nuclear force."

    vecs = model.encode([short, long_same, different])
    v_short, v_long, v_diff = vecs

    print(f"\nEmbedding dimension: {vecs.shape[1]}")
    print(f"\n{'text':<34}{'vector length ||v||':>22}")
    print("-" * 74)
    for label, v in (("short version", v_short), ("same text repeated 8x", v_long),
                     ("unrelated sentence", v_diff)):
        print(f"{label:<34}{np.linalg.norm(v):>22.4f}")

    print(
        "\nNote every length is exactly 1.0000: this model L2-normalizes its output, as most\n"
        "modern embedding models do. That has a consequence worth stating plainly before\n"
        "going further — ON UNIT VECTORS, COSINE AND EUCLIDEAN RANK IDENTICALLY, because\n"
        "\n    ||a - b||^2 = ||a||^2 + ||b||^2 - 2(a.b) = 2 - 2*cos(a,b)\n\n"
        "so euclidean distance is a decreasing function of cosine similarity. Choosing\n"
        "between them here changes nothing at all."
    )

    print(f"\n{'pair':<40}{'cosine':>12}{'euclidean':>14}")
    print("-" * 74)
    for label, v in (("short  vs  same text repeated", v_long),
                     ("short  vs  unrelated sentence", v_diff)):
        print(f"{label:<40}{cosine_similarity(v_short, v):>12.4f}"
              f"{euclidean_distance(v_short, v):>14.4f}")

    lhs = euclidean_distance(v_short, v_diff) ** 2
    rhs = 2 - 2 * cosine_similarity(v_short, v_diff)
    print(f"\nChecking the identity: ||a-b||^2 = {lhs:.8f},  2-2cos = {rhs:.8f}  "
          f"(diff {abs(lhs - rhs):.1e})")

    # So WHY does cosine exist at all? Demonstrate on vectors that are NOT normalized:
    # plain term counts, where repeating a document multiplies every count.
    print(
        "\nSo why does anyone talk about cosine? Because it matters whenever vectors are NOT\n"
        "normalized. Here are the same three texts as raw TERM-COUNT vectors — the kind of\n"
        "representation used before neural embeddings, and still used by BM25 internally:"
    )
    vocab = sorted({w for t in (short, long_same, different) for w in re.findall(r"[a-z]+", t.lower())})
    def counts(t):
        toks = re.findall(r"[a-z]+", t.lower())
        return np.array([toks.count(w) for w in vocab], dtype=float)

    c_short, c_long, c_diff = counts(short), counts(long_same), counts(different)
    print(f"\n{'text':<34}{'vector length ||v||':>22}")
    print("-" * 74)
    for label, c in (("short version", c_short), ("same text repeated 8x", c_long),
                     ("unrelated sentence", c_diff)):
        print(f"{label:<34}{np.linalg.norm(c):>22.4f}")

    print(f"\n{'pair':<40}{'cosine':>12}{'euclidean':>14}")
    print("-" * 74)
    print(f"{'short  vs  same text repeated 8x':<40}{cosine_similarity(c_short, c_long):>12.4f}"
          f"{euclidean_distance(c_short, c_long):>14.4f}")
    print(f"{'short  vs  unrelated sentence':<40}{cosine_similarity(c_short, c_diff):>12.4f}"
          f"{euclidean_distance(c_short, c_diff):>14.4f}")

    cos_same = cosine_similarity(c_short, c_long)
    cos_diff = cosine_similarity(c_short, c_diff)
    euc_same = euclidean_distance(c_short, c_long)
    euc_diff = euclidean_distance(c_short, c_diff)
    verdict = ("EUCLIDEAN GETS IT BACKWARDS" if euc_same > euc_diff
               else "euclidean still orders them correctly, but by a much smaller margin")

    print(
        f"\nNow the two measures disagree, and this is the whole argument for cosine:\n"
        f"  - cosine says the repeated text is {cos_same:.3f} similar to the original — correctly\n"
        f"    identifying it as the SAME CONTENT, because repeating a document scales every\n"
        f"    count by 8 and does not rotate the vector at all.\n"
        f"  - euclidean says it is {euc_same:.2f} away, versus {euc_diff:.2f} for a completely unrelated\n"
        f"    sentence. {verdict}.\n\n"
        f"Euclidean distance is sensitive to vector LENGTH, and for count-based\n"
        f"representations length tracks DOCUMENT LENGTH rather than meaning. Cosine divides\n"
        f"both lengths out and compares direction only: 'what is this about?' instead of\n"
        f"'how much of it is there?'.\n\n"
        f"That is also exactly why modern embedding models normalize their output for you —\n"
        f"once every vector has length 1, the question is settled and a plain dot product is\n"
        f"cosine. Vector databases exploit this: they normalize on insert, then use the\n"
        f"cheapest operation available."
    )


# ---------------------------------------------------------------------------
# Part 2 — do embeddings capture meaning?
# ---------------------------------------------------------------------------


PARAPHRASE_PAIRS = [
    ("How do I reset my password?", "What's the process for changing my login credentials?"),
    ("The restaurant was terrible.", "I had an awful experience dining there."),
    ("Python is a programming language.", "Python is used for writing software."),
    ("The flight was delayed by two hours.", "Our plane departed 120 minutes late."),
]
UNRELATED_PAIRS = [
    ("How do I reset my password?", "The volcano erupted in 1883."),
    ("The restaurant was terrible.", "Quantum entanglement is non-local."),
    ("Python is a programming language.", "She planted tulips in the garden."),
    ("The flight was delayed by two hours.", "Beethoven composed nine symphonies."),
]
# Word overlap is high but meaning is opposite or different — the hard cases.
TRICKY_PAIRS = [
    ("The dog bit the man.", "The man bit the dog."),
    ("I love this product.", "I do not love this product."),
    ("Flight from Paris to Rome", "Flight from Rome to Paris"),
]


def run_meaning_demo(model) -> None:
    print()
    print("=" * 74)
    print("PART 2 — Do embeddings capture meaning, or just word overlap?")
    print("=" * 74)

    def score(pairs):
        out = []
        for a, b in pairs:
            va, vb = model.encode([a, b])
            out.append(cosine_similarity(va, vb))
        return out

    para = score(PARAPHRASE_PAIRS)
    unrel = score(UNRELATED_PAIRS)
    tricky = score(TRICKY_PAIRS)

    print(f"\nPARAPHRASES — same meaning, almost no shared words:\n")
    for (a, b), s in zip(PARAPHRASE_PAIRS, para):
        shared = len(set(a.lower().split()) & set(b.lower().split()))
        print(f"  {s:.3f}  ({shared} shared words)  \"{a[:38]}\" / \"{b[:38]}\"")

    print(f"\nUNRELATED — no meaningful connection:\n")
    for (a, b), s in zip(UNRELATED_PAIRS, unrel):
        print(f"  {s:.3f}  \"{a[:38]}\" / \"{b[:38]}\"")

    print(f"\nmean paraphrase similarity: {np.mean(para):.3f}")
    print(f"mean unrelated similarity:  {np.mean(unrel):.3f}")
    print(f"separation: {np.mean(para) - np.mean(unrel):.3f}")

    print(
        "\nThe paraphrases score far higher despite sharing almost no words — 'reset my\n"
        "password' and 'changing my login credentials' have ONE word in common. A keyword\n"
        "search would rank them as unrelated. This is what 'semantic' means, and it is the\n"
        "entire reason embeddings are worth the trouble."
    )

    print(f"\nNOW THE HARD CASES — high word overlap, different meaning:\n")
    for (a, b), s in zip(TRICKY_PAIRS, tricky):
        print(f"  {s:.3f}  \"{a}\" / \"{b}\"")

    print(
        f"\nThese score {np.mean(tricky):.3f} on average — HIGHER than the genuine paraphrases at "
        f"{np.mean(para):.3f},\neven though two of them mean the opposite of each other.\n\n"
        f"This is a real and widely under-advertised limitation. Sentence embeddings are\n"
        f"largely a bag-of-meaning: they capture topic extremely well and NEGATION and\n"
        f"ARGUMENT ORDER poorly. 'I love this product' and 'I do not love this product' are\n"
        f"about the same thing, and the embedding mostly encodes what it is about.\n\n"
        f"Practical consequences: never use raw cosine similarity as a fact-checker or a\n"
        f"sentiment detector, and in a RAG system expect retrieval to return passages on\n"
        f"the right TOPIC that may contradict the query. Project 16 measures this properly."
    )

    plt.figure(figsize=(7.5, 4.4))
    plt.hist(unrel, bins=np.linspace(-0.2, 1.0, 25), alpha=0.7, label="unrelated")
    plt.hist(para, bins=np.linspace(-0.2, 1.0, 25), alpha=0.7, label="paraphrases")
    plt.hist(tricky, bins=np.linspace(-0.2, 1.0, 25), alpha=0.7,
             label="high overlap, different meaning")
    plt.xlabel("cosine similarity")
    plt.ylabel("count")
    plt.title("Embeddings separate topic well, and negation badly")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "similarity_distributions.png", dpi=120)
    plt.close()
    print("Saved plot to outputs/similarity_distributions.png")


# ---------------------------------------------------------------------------
# A small document collection, used by Parts 3 and 6
# ---------------------------------------------------------------------------

DOCS = [
    "To reset your password, click 'Forgot password' on the sign-in page and follow the emailed link.",
    "Our refund policy allows returns within 30 days of purchase for a full refund.",
    "The annual subscription costs 99 dollars and renews automatically each year.",
    "Two-factor authentication can be enabled from the security settings menu.",
    "Shipping to European addresses typically takes 5 to 7 business days.",
    "You can cancel your subscription at any time from the billing dashboard.",
    "Our support team is available Monday to Friday, 9am to 6pm UTC.",
    "The mobile application is available for both iOS and Android devices.",
    "Data is encrypted at rest using AES-256 and in transit using TLS 1.3.",
    "Enterprise customers receive a dedicated account manager and priority support.",
    "The free tier includes 1000 API calls per month with no credit card required.",
    "We are compliant with GDPR and store European user data within the EU.",
    "Error E404 means the requested resource could not be found on the server.",
    "Error E503 means the service is temporarily unavailable, please retry later.",
    "API version v2.1 deprecates the legacy authentication header.",
    "API version v3.0 introduces streaming responses and webhooks.",
]

QUERIES = [
    ("I forgot my login details", 0),
    ("how do I get my money back", 1),
    ("can I stop paying monthly", 5),
    ("is my information secure", 8),
    ("when can I talk to a human", 6),
    ("how long until my parcel arrives", 4),
]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    """
    Classic keyword search, as the honest baseline embeddings must beat.

    score(q, d) = sum over terms t in q of  IDF(t) * tf-saturation(t, d)        (2)

    It counts words. It cannot know that 'money back' means 'refund' — but it is
    exact, instant, needs no model, and wins whenever the query uses the same words
    as the document (Part 3 shows this happening).
    """

    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [tokenize(d) for d in docs]
        self.k1, self.b = k1, b
        self.avg_len = float(np.mean([len(d) for d in self.docs]))
        self.df = {}
        for d in self.docs:
            for t in set(d):
                self.df[t] = self.df.get(t, 0) + 1
        self.N = len(self.docs)

    def score(self, query: str) -> np.ndarray:
        q = tokenize(query)
        scores = np.zeros(self.N)
        for i, doc in enumerate(self.docs):
            total = 0.0
            for t in q:
                if t not in self.df:
                    continue
                tf = doc.count(t)
                if tf == 0:
                    continue
                idf = math.log(1 + (self.N - self.df[t] + 0.5) / (self.df[t] + 0.5))
                total += idf * (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * len(doc) / self.avg_len))
            scores[i] = total
        return scores


def run_search_comparison(model) -> None:
    print()
    print("=" * 74)
    print("PART 3 — Semantic search vs keyword search: which actually wins?")
    print("=" * 74)

    doc_vecs = model.encode(DOCS, normalize_embeddings=True)
    bm25 = BM25(DOCS)

    print(f"\n{len(DOCS)} support documents, {len(QUERIES)} queries with a known correct answer.\n")
    print(f"{'query':<34}{'BM25 rank':>12}{'embedding rank':>17}")
    print("-" * 74)

    bm25_ranks, emb_ranks = [], []
    for query, correct in QUERIES:
        b_scores = bm25.score(query)
        b_rank = int(np.where(np.argsort(-b_scores) == correct)[0][0]) + 1
        q_vec = model.encode([query], normalize_embeddings=True)[0]
        e_scores = doc_vecs @ q_vec
        e_rank = int(np.where(np.argsort(-e_scores) == correct)[0][0]) + 1
        bm25_ranks.append(b_rank)
        emb_ranks.append(e_rank)
        print(f"{query:<34}{b_rank:>12}{e_rank:>17}")

    def mrr(ranks):
        return float(np.mean([1 / r for r in ranks]))

    print(f"\n{'metric':<26}{'BM25':>10}{'embeddings':>14}")
    print("-" * 74)
    print(f"{'top-1 accuracy':<26}{np.mean([r == 1 for r in bm25_ranks]):>10.3f}"
          f"{np.mean([r == 1 for r in emb_ranks]):>14.3f}")
    print(f"{'mean reciprocal rank':<26}{mrr(bm25_ranks):>10.3f}{mrr(emb_ranks):>14.3f}")

    print(
        "\nMRR (mean reciprocal rank) averages 1/rank of the correct answer: 1.0 means it\n"
        "was always first, 0.5 means typically second. Project 16 covers these metrics.\n\n"
        "Embeddings win here because the queries deliberately avoid the documents' wording\n"
        "— 'get my money back' vs 'refund', 'stop paying monthly' vs 'cancel subscription'.\n"
        "BM25 counts words and cannot bridge that gap.\n\n"
        "BUT DO NOT CONCLUDE THAT KEYWORD SEARCH IS OBSOLETE. Watch what happens with a\n"
        "query that uses exact terminology:"
    )

    # Deliberately hard for embeddings: pairs of documents that are near-identical
    # in topic and differ only in an identifier. Semantic similarity cannot see the
    # difference between E404 and E503; exact term matching can.
    exact_queries = [
        ("E503", 13), ("E404", 12), ("v3.0", 15), ("v2.1", 14),
        ("AES-256", 8), ("GDPR", 11),
    ]
    print(f"\n{'exact-term query':<34}{'BM25 rank':>12}{'embedding rank':>17}")
    print("-" * 74)
    b_exact, e_exact = [], []
    for query, correct in exact_queries:
        b_scores = bm25.score(query)
        b_rank = int(np.where(np.argsort(-b_scores) == correct)[0][0]) + 1
        q_vec = model.encode([query], normalize_embeddings=True)[0]
        e_rank = int(np.where(np.argsort(-(doc_vecs @ q_vec)) == correct)[0][0]) + 1
        b_exact.append(b_rank)
        e_exact.append(e_rank)
        print(f"{query:<34}{b_rank:>12}{e_rank:>17}")

    print(f"\n{'MRR on exact-term queries':<26}{mrr(b_exact):>10.3f}{mrr(e_exact):>14.3f}")
    print(
        f"\nBoth score {mrr(b_exact):.3f} — a TIE, and it is worth being precise about what that does\n"
        f"and does not show. I expected BM25 to WIN here, including on the deliberately\n"
        f"confusable pairs (E404 vs E503, v2.1 vs v3.0), and it did not: with only\n"
        f"{len(DOCS)} documents, one mention of 'E503' is distinctive enough that the embedding\n"
        f"finds it too. The honest conclusion from this experiment is not 'keyword search\n"
        f"wins on identifiers' — it is that EMBEDDINGS LOSE THEIR ADVANTAGE ENTIRELY here.\n\n"
        f"That is still a real result, because the two methods are not equally expensive.\n"
        f"BM25 needs no model, no GPU, no 90 MB download and no embedding step at query\n"
        f"time; it is a word-count table. Paying for a neural model buys you exactly the\n"
        f"{mrr(emb_ranks) - mrr(bm25_ranks):+.3f} MRR on the paraphrased queries above, and nothing on these.\n\n"
        f"Scale changes the picture in BM25's favour: with thousands of near-identical\n"
        f"documents, an embedding genuinely does blur 'E404' into a neighbourhood of\n"
        f"error-message-ish text, while exact term matching stays exact. Exercise 3 builds\n"
        f"that corpus so you can watch it happen rather than take my word for it.\n\n"
        f"Either way the practical answer is the same, and it is why production retrieval is\n"
        f"almost always HYBRID: run both, combine the rankings. Each method covers the\n"
        f"other's failure mode, and on this evidence neither dominates."
    )

    plt.figure(figsize=(7.5, 4.4))
    x = np.arange(2)
    plt.bar(x - 0.18, [mrr(bm25_ranks), mrr(b_exact)], 0.36, label="BM25 (keyword)")
    plt.bar(x + 0.18, [mrr(emb_ranks), mrr(e_exact)], 0.36, label="embeddings (semantic)")
    plt.xticks(x, ["paraphrased queries", "exact-term queries"])
    plt.ylabel("Mean reciprocal rank")
    plt.title("Neither method wins everywhere — which is why hybrid search exists")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "semantic_vs_keyword.png", dpi=120)
    plt.close()
    print("Saved plot to outputs/semantic_vs_keyword.png")


# ---------------------------------------------------------------------------
# Part 4 — the curse of dimensionality
# ---------------------------------------------------------------------------


def run_curse_demo() -> None:
    print()
    print("=" * 74)
    print("PART 4 — The curse of dimensionality, measured")
    print("=" * 74)

    print("\nDraw 1000 random points in d dimensions and look at the distances between them:\n")
    print(f"{'d':>6}{'mean distance':>16}{'std':>10}{'(max-min)/min':>17}")
    print("-" * 74)
    dims, contrasts = [], []
    for d in (2, 8, 32, 128, 512, 2048):
        pts = RNG.normal(size=(1000, d))
        q = RNG.normal(size=d)
        dists = np.linalg.norm(pts - q, axis=1)
        contrast = (dists.max() - dists.min()) / dists.min()
        dims.append(d)
        contrasts.append(contrast)
        print(f"{d:>6}{dists.mean():>16.3f}{dists.std():>10.3f}{contrast:>17.3f}")

    print(
        "\nRead the last column: it is the gap between the nearest and furthest point,\n"
        "relative to the nearest. In 2 dimensions the furthest point is many times further\n"
        "than the nearest. By 2048 dimensions they are nearly the SAME DISTANCE.\n\n"
        "That is the curse of dimensionality: in high dimensions, everything is roughly\n"
        "equidistant from everything else, so 'nearest neighbour' becomes a weak notion and\n"
        "distance-based methods lose their grip. Note the mean distance also grows like\n"
        "sqrt(d), which is why raw distance values are not comparable across dimensions.\n\n"
        "Why embeddings still work despite this: real embeddings do NOT fill their space\n"
        "uniformly like these random points. They lie on a much lower-dimensional surface\n"
        "inside it (the 'manifold hypothesis'), because real text has far less variety than\n"
        "random noise. The curse is a warning about worst cases, not a proof that 384\n"
        "dimensions cannot work — Part 3 just showed them working."
    )

    plt.figure(figsize=(7.5, 4.4))
    plt.plot(dims, contrasts, marker="o")
    plt.xscale("log")
    plt.xlabel("dimensions (log scale)")
    plt.ylabel("(max distance - min distance) / min distance")
    plt.title("In high dimensions, near and far stop being distinguishable")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "curse_of_dimensionality.png", dpi=120)
    plt.close()
    print("Saved plot to outputs/curse_of_dimensionality.png")


# ---------------------------------------------------------------------------
# Part 5 — brute force vs an approximate index
# ---------------------------------------------------------------------------


class IVFIndex:
    """
    A miniature inverted-file index — the idea behind FAISS's IVF and most vector
    databases:

        1. cluster the vectors into n_clusters groups (project 05's k-means)
        2. at query time, find the nearest few CENTROIDS
        3. search only the vectors in those clusters                            (3)

    You skip most of the database, so it is fast; you may miss a neighbour sitting
    just across a cluster boundary, so it is APPROXIMATE. n_probe controls the
    tradeoff directly.
    """

    def __init__(self, vectors, n_clusters=64, seed=0):
        from sklearn.cluster import KMeans
        self.vectors = vectors
        self.km = KMeans(n_clusters=n_clusters, n_init=3, random_state=seed).fit(vectors)
        self.assign = self.km.labels_
        self.buckets = {c: np.where(self.assign == c)[0] for c in range(n_clusters)}

    def search(self, q, k=10, n_probe=4):
        centroid_sim = self.km.cluster_centers_ @ q
        probes = np.argsort(-centroid_sim)[:n_probe]
        candidates = np.concatenate([self.buckets[c] for c in probes if len(self.buckets[c])])
        if len(candidates) == 0:
            return np.array([], dtype=int)
        sims = self.vectors[candidates] @ q
        return candidates[np.argsort(-sims)[:k]]


def run_index_demo() -> None:
    print()
    print("=" * 74)
    print("PART 5 — Brute force vs an approximate index: speed against recall")
    print("=" * 74)

    n, d, k = 50_000, 128, 10
    vectors = RNG.normal(size=(n, d)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    queries = vectors[RNG.choice(n, 200, replace=False)] + 0.35 * RNG.normal(size=(200, d))
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    print(f"\n{n:,} vectors of {d} dimensions, {len(queries)} queries, top-{k} retrieval.\n")

    t0 = time.perf_counter()
    truth = [np.argsort(-(vectors @ q))[:k] for q in queries]
    brute_time = time.perf_counter() - t0

    index = IVFIndex(vectors, n_clusters=64)
    print(f"{'method':<28}{'time (s)':>11}{'speedup':>10}{'recall@10':>12}")
    print("-" * 74)
    print(f"{'brute force (exact)':<28}{brute_time:>11.3f}{1.0:>10.1f}x{1.000:>12.3f}")

    probes, recalls, speeds = [], [], []
    for n_probe in (1, 2, 4, 8, 16, 32):
        t0 = time.perf_counter()
        got = [index.search(q, k=k, n_probe=n_probe) for q in queries]
        t = time.perf_counter() - t0
        recall = float(np.mean([len(set(a) & set(b)) / k for a, b in zip(got, truth)]))
        probes.append(n_probe)
        recalls.append(recall)
        speeds.append(brute_time / t)
        print(f"{f'IVF index, n_probe={n_probe}':<28}{t:>11.3f}{brute_time / t:>10.1f}x{recall:>12.3f}")

    print(
        "\nRecall@10 is the fraction of the true top-10 that the index actually returned.\n"
        "Every row is the same index — only how many clusters it inspects changes.\n\n"
        "This is the tradeoff every vector database exposes, under various names. Brute\n"
        "force is exact and scales linearly with the collection: fine for thousands of\n"
        "vectors, hopeless for a hundred million. An approximate index trades a few percent\n"
        "of recall for one or two orders of magnitude of speed.\n\n"
        "Note that these are RANDOM vectors, which is the hardest possible case for an\n"
        "index — there is no cluster structure to exploit, so recall at low n_probe is\n"
        "worse than you would see on real embeddings, which do cluster by topic. Real\n"
        "systems use HNSW (a navigable graph) more often than IVF, but the shape of the\n"
        "tradeoff is identical."
    )

    plt.figure(figsize=(7.5, 4.4))
    plt.plot(speeds, recalls, marker="o")
    for s, r, p in zip(speeds, recalls, probes):
        plt.annotate(f"n_probe={p}", (s, r), fontsize=7, xytext=(4, -8),
                     textcoords="offset points")
    plt.axhline(1.0, color="grey", linestyle=":", label="brute force (exact, 1.0x)")
    plt.xlabel("Speedup over brute force")
    plt.ylabel("Recall@10")
    plt.title("Approximate search: every point is a choice of speed vs completeness")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "index_tradeoff.png", dpi=120)
    plt.close()
    print("Saved plot to outputs/index_tradeoff.png")


# ---------------------------------------------------------------------------
# Part 6 — chunking
# ---------------------------------------------------------------------------


ARTICLE = """
The company was founded in 2015 by three engineers who had previously worked together
at a large search company. Their stated goal was to make enterprise data searchable
without requiring customers to migrate it into a new system. The first product shipped
in March 2017 to eleven design partners.
Revenue grew steadily through 2019, reaching twelve million dollars annually. The
pricing model was seat-based until 2018, when it changed to consumption-based billing
after customers complained that seats penalised automation.
In 2020 the company raised a Series B round of forty million dollars led by an
infrastructure-focused fund based in Boston. The round valued the business at three
hundred million dollars post-money. An earlier Series A of nine million had closed in
2018 at a fifty million valuation.
Following the raise, headcount doubled from ninety to one hundred and eighty people
over eighteen months. Most of that growth was in engineering and customer support,
though the company also opened its first office outside the United States, in Lisbon,
staffed initially by four people.
The 2021 outage was the most serious incident in the company's history. A misconfigured
database migration caused eleven hours of downtime affecting roughly forty percent of
customers. The post-mortem identified inadequate staging environments as the root cause,
and the engineering team subsequently rebuilt the deployment pipeline around immutable
infrastructure and automated rollback.
Compliance work began in earnest in 2022. The company achieved SOC 2 Type II
certification in June and ISO 27001 the following January. Data residency options were
added for European and Australian customers, and the security team grew from two people
to seven over the same period.
The machine learning team was formed in 2023 with a mandate to improve search ranking.
Their first shipped model reduced median query latency by thirty-one percent while
improving click-through on the top result. A second project, automatic query expansion,
was cancelled after six months when offline gains failed to reproduce in live traffic.
"""


def chunk_text(text: str, size: int, overlap: int = 0) -> list[str]:
    """Split into `size`-word chunks with `overlap` words repeated between them."""
    words = text.split()
    step = max(size - overlap, 1)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step)
            if words[i:i + size]]


def run_chunking_demo(model) -> None:
    print()
    print("=" * 74)
    print("PART 6 — Chunking: the boring decision that decides whether RAG works")
    print("=" * 74)

    questions = [
        ("What caused the 2021 outage?", "misconfigured database migration"),
        ("How much did the Series B raise?", "forty million"),
        ("How many people work at the company?", "one hundred and eighty"),
        ("When did the company get SOC 2 certified?", "soc 2 type ii"),
        ("Why was query expansion cancelled?", "failed to reproduce"),
        ("What was the Series A valuation?", "fifty million"),
        ("Where is the non-US office?", "lisbon"),
        ("Why did pricing change from seats?", "penalised automation"),
        ("How much did latency improve?", "thirty-one percent"),
    ]

    print(f"\nOne article, chunked several ways, then queried. A retrieval counts as correct")
    print(f"if the top-1 chunk contains the answer text.\n")
    print(f"{'chunk size (words)':<22}{'overlap':>9}{'chunks':>9}{'top-1 correct':>16}")
    print("-" * 74)

    configs = [(8, 0), (15, 0), (30, 0), (60, 0), (120, 0), (240, 0),
               (30, 10), (60, 20), (120, 40)]
    results = []
    for size, overlap in configs:
        chunks = chunk_text(ARTICLE, size, overlap)
        vecs = model.encode(chunks, normalize_embeddings=True)
        correct = 0
        for q, answer in questions:
            qv = model.encode([q], normalize_embeddings=True)[0]
            best = chunks[int(np.argmax(vecs @ qv))]
            correct += answer.lower() in best.lower()
        results.append((size, overlap, len(chunks), correct / len(questions)))
        print(f"{size:<22}{overlap:>9}{len(chunks):>9}{correct / len(questions):>16.3f}")

    print(
        "\nSame article, same model, same questions — only the chunking changed, and it\n"
        "swings top-1 accuracy from 0.222 to 1.000. That is a bigger range than you would\n"
        "get by switching embedding models.\n\n"
        "TOO SMALL clearly fails, and the trend is unambiguous: 8 words scores 0.222, and\n"
        "accuracy climbs steadily as chunks grow. The reason is that the answer gets\n"
        "separated from the context that makes it findable — an 8-word chunk may contain\n"
        "'forty million dollars' with nothing indicating it refers to the Series B, so the\n"
        "query cannot match it.\n\n"
        "OVERLAP is the cleanest result here. Every overlapping configuration beats its\n"
        "non-overlapping counterpart at the same chunk size: 0.556 -> 0.889 at 30 words,\n"
        "0.778 -> 1.000 at 60, 0.667 -> 0.889 at 120. Facts that straddle a boundary would\n"
        "otherwise be cut in half and lost from both chunks; repeating a few words at each\n"
        "join fixes that. The cost is storing and searching more chunks.\n\n"
        "TOO LARGE is the claim this experiment does NOT establish, and it would be easy to\n"
        "pretend otherwise. The textbook argument is that big chunks dilute the signal by\n"
        "averaging several topics into one vector, and there is a hint of it at 120 words\n"
        "(0.667, a dip below 60 words) — but 240 words recovers to 0.778, so there is no\n"
        "clean decline. The likely reason is that this article is only ~450 words, so a\n"
        "240-word chunk is half the document and there is nowhere further to go. Testing\n"
        "the dilution claim properly needs a much longer document; exercise 5 does that.\n\n"
        "What you can take away: there is no universally right chunk size, small is clearly\n"
        "bad, overlap reliably helps, and this unglamorous preprocessing decision moves RAG\n"
        "quality more than most of the choices people spend their time on. Project 13\n"
        "builds the full pipeline on top of exactly this."
    )

    plt.figure(figsize=(7.5, 4.4))
    labels = [f"{s}w\n+{o}" for s, o, _, _ in results]
    plt.bar(labels, [r[3] for r in results], color="steelblue")
    plt.ylabel("Top-1 retrieval accuracy")
    plt.xlabel("chunk size (words) + overlap")
    plt.title("Chunking changes retrieval accuracy on identical content")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "chunking.png", dpi=120)
    plt.close()
    print("Saved plot to outputs/chunking.png")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("Loading the embedding model (downloads ~90 MB on first run)...\n")
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    run_similarity_demo(embedder)
    run_meaning_demo(embedder)
    run_search_comparison(embedder)
    run_curse_demo()
    run_index_demo()
    run_chunking_demo(embedder)
