"""
RAG from Scratch — no framework, every stage written out. Six experiments:

  Part 1  the problem RAG solves: an LLM asked about facts it cannot know
  Part 2  the pipeline, stage by stage, with the actual prompt printed
  Part 3  retrieval quality caps answer quality — including confident wrong answers
  Part 4  how many chunks to retrieve? more is not better
  Part 5  hybrid retrieval: dense + BM25 (project 12's two methods, combined)
  Part 6  abstention: does it admit when the answer is not in the documents?

Uses GROQ_API_KEY from ../.env for GENERATION only; retrieval is local
(sentence-transformers). Runs without a key — generation parts skip with a message.

LLM responses are cached in ./cache so re-runs are free and deterministic.

Run:
    python rag_from_scratch.py
"""

import hashlib
import json
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

HERE = Path(__file__).parent
OUTPUT_DIR = HERE / "outputs"
CACHE_DIR = HERE / "cache"
MODEL = "llama-3.1-8b-instant"


# ---------------------------------------------------------------------------
# The knowledge base — deliberately FICTIONAL
# ---------------------------------------------------------------------------
#
# Every fact below is invented. That is the point: if the documents described a
# real company, the model might answer from pretraining and we could not tell
# retrieval apart from recall. With invented facts, a correct answer PROVES the
# information came from the retrieved context.

HANDBOOK = """
Meridian Analytics was founded in 2019 in Tallinn by Ilse Rauda and Petr Kovac.
The company builds observability tooling for data pipelines.

Employees are entitled to 28 days of paid annual leave per calendar year, plus
public holidays. Leave requests must be submitted at least 14 days in advance
through the Zephyr portal. Unused leave may be carried over, up to a maximum of
5 days, and expires on 31 March of the following year.

The company operates a remote-first policy. Employees may work from any country
in which Meridian has a legal entity: Estonia, Portugal, Canada and Singapore.
Working from a country outside that list requires written approval from the
People team and is limited to 45 days per year.

The annual learning budget is 1,800 euros per employee. It covers conferences,
books, courses and certifications. Hardware is not covered by the learning budget
and is requested separately through the IT team.

Meridian's core product is called Lumen. Lumen version 4.2, released in March
2024, introduced automatic anomaly detection for streaming pipelines. The
previous version, 4.1, had added support for the Iceberg table format.

Production incidents are graded from SEV1 to SEV4. A SEV1 means complete service
unavailability and pages the on-call engineer immediately. A SEV2 means degraded
service for more than 20 percent of customers. Post-mortems are required for all
SEV1 and SEV2 incidents and must be published within 5 working days.

The on-call rotation is weekly and runs from Tuesday 10:00 UTC to the following
Tuesday 10:00 UTC. Engineers on call receive a stipend of 400 euros per rotation.
No engineer may be scheduled for more than one rotation in any four-week period.

New employees complete a 90-day onboarding programme. The first two weeks are
dedicated to systems training, after which each new engineer ships a small change
to production, traditionally on day 12. Probation reviews occur at day 90.

Expenses under 200 euros may be approved by any team lead. Expenses between 200
and 2,000 euros require director approval. Anything above 2,000 euros requires
sign-off from the finance team and one of the founders.

The company uses a 4-day working week during July and August. Salaries are
unaffected. This policy was introduced in 2022 after a six-month trial.
"""

# Questions whose answers ARE in the handbook. The expected string is used for
# exact grading, so no human or LLM judgement is involved.
QUESTIONS = [
    ("How many days of annual leave do employees get?", "28"),
    ("How much is the annual learning budget?", "1,800"),
    ("Who founded Meridian Analytics?", "Rauda"),
    ("What is the on-call stipend per rotation?", "400"),
    ("Which version of Lumen added Iceberg support?", "4.1"),
    ("How long is the onboarding programme?", "90"),
    ("How many days of leave can be carried over?", "5"),
    ("What approval is needed for a 900 euro expense?", "director"),
]

# Questions that sound plausible but are NOT answerable from the handbook.
# Used in Part 6 to measure whether the system admits ignorance or invents.
UNANSWERABLE = [
    "What is Meridian's parental leave policy?",
    "How many employees does Meridian have?",
    "What is the company's revenue?",
    "Does Meridian offer a pension scheme?",
    "What programming language is Lumen written in?",
    "Who is the current CTO?",
]

# Harder: also unanswerable, but a plausible WRONG number sits in the retrieved
# context and can be mistaken for the answer. The second element is the decoy the
# model must NOT report. This is the realistic RAG failure — not a missing topic,
# but an adjacent one that looks close enough.
TRAPS = [
    ("What is the annual hardware budget per employee?", "1,800"),
    ("How many days of paid sick leave do employees get?", "28"),
    ("What is the bonus for resolving a SEV1 incident?", "400"),
    ("How long is the probation period for contractors?", "90"),
    ("How many days of parental leave are granted?", "28"),
]


# ---------------------------------------------------------------------------
# The LLM client — cached, and optional
# ---------------------------------------------------------------------------


class LLM:
    """
    A thin wrapper over Groq's OpenAI-compatible endpoint, with a disk cache.

    Caching matters for more than speed: it makes the whole experiment
    REPRODUCIBLE. At temperature 0 the same prompt returns the same answer, so
    re-running the script gives identical numbers instead of numbers that drift.
    """

    def __init__(self):
        CACHE_DIR.mkdir(exist_ok=True)
        self.available = False
        self.calls = 0
        self.cached = 0
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            return
        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
            self.available = True
        except ImportError:
            pass

    def ask(self, prompt: str, system: str = "", max_tokens: int = 200) -> str:
        digest = hashlib.sha256(
            f"{MODEL}|{system}|{prompt}|{max_tokens}".encode()).hexdigest()[:24]
        path = CACHE_DIR / f"{digest}.json"
        if path.exists():
            self.cached += 1
            return json.loads(path.read_text())["response"]
        if not self.available:
            return ""

        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
        for attempt in range(4):
            try:
                r = self.client.chat.completions.create(
                    model=MODEL, messages=messages, max_tokens=max_tokens, temperature=0)
                text = (r.choices[0].message.content or "").strip()
                path.write_text(json.dumps({"prompt": prompt, "response": text}))
                self.calls += 1
                return text
            except Exception as exc:  # rate limit or transient failure
                if attempt == 3:
                    print(f"    (LLM call failed: {type(exc).__name__})")
                    return ""
                time.sleep(2 * (attempt + 1))
        return ""


def graded(response: str, expected: str) -> bool:
    """Exact, deterministic grading: does the expected fact appear in the answer?"""
    norm = re.sub(r"[,\s]+", "", response.lower())
    return re.sub(r"[,\s]+", "", expected.lower()) in norm


# ---------------------------------------------------------------------------
# The retriever — projects 12's two methods
# ---------------------------------------------------------------------------


def chunk_text(text: str, size: int = 60, overlap: int = 20) -> list[str]:
    """
    Split into overlapping word windows.                                       (1)

    Size and overlap come straight from project 12's Part 6, which measured
    overlap helping at every chunk size tested.
    """
    words = text.split()
    step = max(size - overlap, 1)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step)
            if words[i:i + size]]


def tokenize(t: str) -> list[str]:
    return re.findall(r"[a-z0-9.]+", t.lower())


class Retriever:
    """
    Dense (embedding) retrieval, BM25 keyword retrieval, and their fusion.

    dense:  score(q, c) = cos(embed(q), embed(c))                              (2)
    BM25:   project 12's formula (3)
    hybrid: reciprocal rank fusion, RRF(c) = sum_r 1 / (k + rank_r(c))         (4)

    RRF combines RANKINGS rather than scores, which matters because a cosine
    similarity of 0.7 and a BM25 score of 12.4 are not on comparable scales.
    """

    def __init__(self, chunks, embedder):
        self.chunks = chunks
        self.embedder = embedder
        self.vecs = embedder.encode(chunks, normalize_embeddings=True)
        self.tok = [tokenize(c) for c in chunks]
        self.avg_len = float(np.mean([len(t) for t in self.tok]))
        self.df = {}
        for t in self.tok:
            for w in set(t):
                self.df[w] = self.df.get(w, 0) + 1

    def dense_scores(self, query: str) -> np.ndarray:
        q = self.embedder.encode([query], normalize_embeddings=True)[0]
        return self.vecs @ q  # (2)

    def bm25_scores(self, query: str, k1=1.5, b=0.75) -> np.ndarray:
        q = tokenize(query)
        n = len(self.chunks)
        out = np.zeros(n)
        for i, doc in enumerate(self.tok):
            s = 0.0
            for w in q:
                if w not in self.df:
                    continue
                tf = doc.count(w)
                if tf == 0:
                    continue
                idf = math.log(1 + (n - self.df[w] + 0.5) / (self.df[w] + 0.5))
                s += idf * tf * (k1 + 1) / (
                    tf + k1 * (1 - b + b * len(doc) / self.avg_len))
            out[i] = s
        return out

    def retrieve(self, query: str, k: int = 3, method: str = "dense") -> list[int]:
        if method == "dense":
            return list(np.argsort(-self.dense_scores(query))[:k])
        if method == "bm25":
            return list(np.argsort(-self.bm25_scores(query))[:k])
        # hybrid — reciprocal rank fusion (4)
        d_rank = {c: r for r, c in enumerate(np.argsort(-self.dense_scores(query)))}
        b_rank = {c: r for r, c in enumerate(np.argsort(-self.bm25_scores(query)))}
        rrf = {c: 1 / (60 + d_rank[c]) + 1 / (60 + b_rank[c]) for c in range(len(self.chunks))}
        return sorted(rrf, key=rrf.get, reverse=True)[:k]


PROMPT = """Answer the question using ONLY the context below.

Context:
{context}

Question: {question}
Answer concisely."""

STRICT_PROMPT = """Answer the question using ONLY the context below.
If the context does not contain the answer, reply exactly: NOT IN CONTEXT

Context:
{context}

Question: {question}
Answer concisely."""


def build_prompt(chunks, indices, question, strict=False) -> str:
    context = "\n\n".join(f"[{i}] {chunks[i]}" for i in indices)
    return (STRICT_PROMPT if strict else PROMPT).format(context=context, question=question)


# ---------------------------------------------------------------------------
# Part 1 — the problem
# ---------------------------------------------------------------------------


def run_problem_demo(llm, retriever, chunks) -> None:
    print("=" * 74)
    print("PART 1 — The problem: an LLM asked about facts it cannot possibly know")
    print("=" * 74)

    print("\nEvery fact in this handbook is INVENTED. That is deliberate: if the company")
    print("were real, the model might answer from pretraining and we could not tell")
    print("retrieval apart from recall. Here, a correct answer PROVES the information")
    print("came from the context we supplied.\n")
    print(f"{'question':<48}{'no context':>13}{'with RAG':>11}")
    print("-" * 74)

    no_ctx, with_ctx = [], []
    for q, expected in QUESTIONS:
        bare = llm.ask(q, max_tokens=120)
        idx = retriever.retrieve(q, k=3)
        rag = llm.ask(build_prompt(chunks, idx, q), max_tokens=120)
        ok_bare, ok_rag = graded(bare, expected), graded(rag, expected)
        no_ctx.append(ok_bare)
        with_ctx.append(ok_rag)
        print(f"{q[:46]:<48}{'YES' if ok_bare else 'no':>13}{'YES' if ok_rag else 'no':>11}")

    print(f"\n{'accuracy':<48}{np.mean(no_ctx):>13.3f}{np.mean(with_ctx):>11.3f}")
    print(
        f"\nWithout context the model scores {np.mean(no_ctx):.1%} — it has never seen this handbook,\n"
        f"so it cannot know. With three retrieved chunks it scores {np.mean(with_ctx):.1%}.\n\n"
        f"That gap is the entire value of RAG, and note what it is NOT: the model did not\n"
        f"become smarter, and nothing was retrained. We simply put the right text in front\n"
        f"of it. RAG is a retrieval problem wearing a generation costume.\n\n"
        f"Read the 'no context' answers yourself in the cache directory — the failure mode\n"
        f"is worth seeing. The model does not usually say 'I don't know'; it produces a\n"
        f"fluent, plausible, wrong answer. That is hallucination, and it is what makes\n"
        f"grounding necessary rather than merely convenient."
    )


# ---------------------------------------------------------------------------
# Part 2 — the pipeline
# ---------------------------------------------------------------------------


def run_pipeline_demo(llm, retriever, chunks) -> None:
    print()
    print("=" * 74)
    print("PART 2 — The pipeline, stage by stage")
    print("=" * 74)

    question = "How many days of leave can be carried over, and when do they expire?"
    print(f"\nQuestion: \"{question}\"\n")

    print("STAGE 1 — CHUNK.  The handbook is split into overlapping word windows.")
    print(f"  {len(HANDBOOK.split())} words -> {len(chunks)} chunks of ~60 words with 20 words of overlap\n")

    print("STAGE 2 — EMBED & RETRIEVE.  Score every chunk, keep the best.")
    scores = retriever.dense_scores(question)
    order = np.argsort(-scores)
    for rank, i in enumerate(order[:5]):
        marker = "<- retrieved" if rank < 3 else ""
        print(f"  rank {rank + 1}  cos={scores[i]:.3f}  chunk {i:>2}: "
              f"\"{chunks[i][:52]}...\" {marker}")

    idx = list(order[:3])
    prompt = build_prompt(chunks, idx, question)
    print(f"\nSTAGE 3 — AUGMENT.  The retrieved text is pasted into a prompt:\n")
    print("  " + "\n  ".join(prompt[:420].split("\n")) + "\n  ...")

    answer = llm.ask(prompt, max_tokens=150)
    print(f"\nSTAGE 4 — GENERATE.\n")
    print(f"  {answer}")

    print(
        "\nThat is all RAG is: four stages, no framework, about 40 lines of code. Project 14\n"
        "rebuilds exactly this in LangChain, and the comparison is the point — you will be\n"
        "able to see precisely which of these four stages each abstraction is hiding.\n\n"
        "Notice the model never saw the handbook during training and has no memory between\n"
        "calls. Everything it knows about Meridian arrived in the prompt, and will be gone\n"
        "the moment the call returns."
    )


# ---------------------------------------------------------------------------
# Part 3 — retrieval quality caps answer quality
# ---------------------------------------------------------------------------


def run_retrieval_quality_demo(llm, retriever, chunks) -> None:
    print()
    print("=" * 74)
    print("PART 3 — Retrieval quality caps answer quality")
    print("=" * 74)

    print("\nSame model, same questions, three different contexts:")
    print("  correct  — the chunks the retriever actually selected")
    print("  wrong    — the LOWEST-scoring chunks, i.e. deliberately irrelevant text")
    print("  none     — no context at all\n")
    print(f"{'question':<44}{'correct':>10}{'wrong':>9}{'none':>8}")
    print("-" * 74)

    results = {"correct": [], "wrong": [], "none": []}
    wrong_answers = []

    def abstained(text: str) -> bool:
        t = text.lower()
        return any(p in t for p in ("not in context", "does not contain", "doesn\'t contain",
                                    "not mentioned", "no information", "not specified",
                                    "not provided", "cannot answer", "does not provide"))
    for q, expected in QUESTIONS:
        scores = retriever.dense_scores(q)
        good = list(np.argsort(-scores)[:3])
        bad = list(np.argsort(scores)[:3])  # the least relevant chunks

        a_good = llm.ask(build_prompt(chunks, good, q), max_tokens=120)
        a_bad = llm.ask(build_prompt(chunks, bad, q), max_tokens=120)
        a_none = llm.ask(q, max_tokens=120)

        for key, ans in (("correct", a_good), ("wrong", a_bad), ("none", a_none)):
            results[key].append(graded(ans, expected))
        if not graded(a_bad, expected) and a_bad:
            wrong_answers.append((q, a_bad, abstained(a_bad)))
        print(f"{q[:42]:<44}{'YES' if results['correct'][-1] else 'no':>10}"
              f"{'YES' if results['wrong'][-1] else 'no':>9}"
              f"{'YES' if results['none'][-1] else 'no':>8}")

    print(f"\n{'accuracy':<44}{np.mean(results['correct']):>10.3f}"
          f"{np.mean(results['wrong']):>9.3f}{np.mean(results['none']):>8.3f}")

    print(
        f"\nThe generator can only be as good as what it is given. With irrelevant chunks it\n"
        f"scores {np.mean(results['wrong']):.1%} — about the same as no context at all. No amount of prompt\n"
        f"engineering fixes a retrieval failure, because the information simply is not there.\n\n"
        f"THIS IS THE MOST IMPORTANT THING IN THE PROJECT: when RAG fails, it usually fails\n"
        f"at retrieval, and people spend their time tuning the prompt. Measure retrieval\n"
        f"separately (project 16) before touching the generation step."
    )

    n_abstain = sum(1 for _, _, ab in wrong_answers if ab)
    if wrong_answers:
        q, a, ab = wrong_answers[0]
        print(f"\nWhat does a failure actually look like? Of the {len(wrong_answers)} wrong-context answers,")
        print(f"{n_abstain} correctly said the context did not contain the answer:\n")
        print(f"  Q: {q}")
        print(f"  A: {a[:200]}")
        print(
            f"\nThat is better behaviour than I expected, and worth stating plainly rather than\n"
            f"quietly dropping: given context that is CLEARLY irrelevant, this model mostly\n"
            f"refuses instead of inventing. The dangerous case is not obviously-wrong context —\n"
            f"it is context that looks close enough to be mistaken for an answer. Part 6\n"
            f"constructs exactly that and measures it."
        )

    plt.figure(figsize=(7.5, 4.4))
    labels = ["correct context", "wrong context", "no context"]
    vals = [np.mean(results["correct"]), np.mean(results["wrong"]), np.mean(results["none"])]
    plt.bar(labels, vals, color=["seagreen", "indianred", "grey"])
    for i, v in enumerate(vals):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
    plt.ylabel("Answer accuracy")
    plt.ylim(0, 1.1)
    plt.title("The generator cannot exceed the retriever")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "retrieval_quality.png", dpi=120)
    plt.close()
    print("\nSaved plot to outputs/retrieval_quality.png")


# ---------------------------------------------------------------------------
# Part 4 — how many chunks?
# ---------------------------------------------------------------------------


def run_topk_demo(llm, retriever, chunks) -> None:
    print()
    print("=" * 74)
    print("PART 4 — How many chunks should you retrieve?")
    print("=" * 74)

    ks = [1, 2, 3, 5, 8, 12]
    accs, ctx_words = [], []
    print(f"\n{'k (chunks retrieved)':>22}{'answer accuracy':>18}{'context words':>16}")
    print("-" * 74)
    for k in ks:
        correct, words = [], []
        for q, expected in QUESTIONS:
            idx = retriever.retrieve(q, k=k)
            prompt = build_prompt(chunks, idx, q)
            words.append(sum(len(chunks[i].split()) for i in idx))
            correct.append(graded(llm.ask(prompt, max_tokens=120), expected))
        accs.append(float(np.mean(correct)))
        ctx_words.append(float(np.mean(words)))
        print(f"{k:>22}{accs[-1]:>18.3f}{ctx_words[-1]:>16.0f}")

    best_k = ks[int(np.argmax(accs))]
    print(
        f"\nAccuracy peaks at k={best_k} ({max(accs):.3f}). Retrieving MORE is not simply better,\n"
        f"and there are two costs pulling in opposite directions:\n"
        f"  - too FEW chunks and the answer may not be in the context at all;\n"
        f"  - too MANY and the relevant sentence is buried among distractors, while every\n"
        f"    extra chunk costs tokens (and therefore money and latency) on every query.\n\n"
        f"Context words grew from {ctx_words[0]:.0f} at k=1 to {ctx_words[-1]:.0f} at k={ks[-1]} — a "
        f"{ctx_words[-1] / ctx_words[0]:.0f}x increase in cost for no gain\nat all in accuracy.\n\n"
        f"BE PRECISE ABOUT WHAT THIS DOES AND DOES NOT SHOW. The cost side is demonstrated:\n"
        f"more chunks means strictly more tokens on every query, forever. The DISTRACTION\n"
        f"side is not — accuracy holds at {accs[-1]:.3f} even at k={ks[-1]}, so on this corpus the model is\n"
        f"not measurably confused by extra context. The reason is that the whole handbook is\n"
        f"only {len(HANDBOOK.split())} words, so k=12 retrieves essentially the entire document and the answer\n"
        f"is always present.\n\n"
        f"The 'lost in the middle' effect is real in the literature, but this experiment is\n"
        f"far too small to show it, and it would be dishonest to present the token cost as\n"
        f"if it were evidence of it. Exercise 4 scales the corpus up so the effect can\n"
        f"actually be tested."
    )

    fig, ax1 = plt.subplots(figsize=(7.5, 4.4))
    ax1.plot(ks, accs, marker="o", color="seagreen", label="answer accuracy")
    ax1.set_xlabel("k — number of chunks retrieved")
    ax1.set_ylabel("Answer accuracy", color="seagreen")
    ax1.set_ylim(0, 1.05)
    ax2 = ax1.twinx()
    ax2.plot(ks, ctx_words, marker="s", color="indianred", label="context size")
    ax2.set_ylabel("Context words sent per query", color="indianred")
    plt.title("More retrieved chunks costs tokens; accuracy does not keep rising")
    fig.tight_layout()
    plt.savefig(OUTPUT_DIR / "topk.png", dpi=120)
    plt.close()
    print("Saved plot to outputs/topk.png")


# ---------------------------------------------------------------------------
# Part 5 — hybrid retrieval
# ---------------------------------------------------------------------------


def run_hybrid_demo(llm, retriever, chunks) -> None:
    print()
    print("=" * 74)
    print("PART 5 — Hybrid retrieval: dense + BM25")
    print("=" * 74)

    # Retrieval quality on its own terms: is the answer text present in the context?
    print("\nFirst, retrieval measured WITHOUT the LLM — does the retrieved context even")
    print("contain the answer string? This separates retrieval failure from generation")
    print("failure, which Part 3 showed is the distinction that matters.\n")
    print(f"{'method':<14}{'recall@3 (answer in context)':>32}{'answer accuracy':>18}")
    print("-" * 74)

    summary = {}
    for method in ("dense", "bm25", "hybrid"):
        recall, correct = [], []
        for q, expected in QUESTIONS:
            idx = retriever.retrieve(q, k=3, method=method)
            context = " ".join(chunks[i] for i in idx)
            recall.append(graded(context, expected))
            correct.append(graded(llm.ask(build_prompt(chunks, idx, q), max_tokens=120),
                                  expected))
        summary[method] = (float(np.mean(recall)), float(np.mean(correct)))
        print(f"{method:<14}{summary[method][0]:>32.3f}{summary[method][1]:>18.3f}")

    best = max(summary, key=lambda m: summary[m][1])
    print(
        f"\nRecall@3 is the honest upper bound on answer accuracy: if the answer string is\n"
        f"not in the retrieved context, no generator can produce it except by luck or by\n"
        f"hallucinating something that happens to be right.\n\n"
        f"Best method here: {best}. Note the numbers are close and there are only "
        f"{len(QUESTIONS)} questions,\nso this is weak evidence — do not conclude that one method is "
        f"generally superior\nfrom eight queries on one document. Project 12 measured the same "
        f"comparison more\ncarefully and found dense winning on paraphrased queries and tying on "
        f"exact terms.\n\n"
        f"The argument for hybrid is not that it always wins; it is that it rarely loses,\n"
        f"because the two methods fail on different queries. RRF combines RANKINGS rather\n"
        f"than scores precisely so that a cosine of 0.7 and a BM25 score of 12.4 never have\n"
        f"to be made comparable."
    )

    plt.figure(figsize=(7.5, 4.4))
    x = np.arange(3)
    plt.bar(x - 0.18, [summary[m][0] for m in ("dense", "bm25", "hybrid")], 0.36,
            label="recall@3 (answer present)")
    plt.bar(x + 0.18, [summary[m][1] for m in ("dense", "bm25", "hybrid")], 0.36,
            label="answer accuracy")
    plt.xticks(x, ["dense", "BM25", "hybrid (RRF)"])
    plt.ylim(0, 1.15)
    plt.ylabel("score")
    plt.title("Retrieval sets the ceiling; generation lives underneath it")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "hybrid.png", dpi=120)
    plt.close()
    print("Saved plot to outputs/hybrid.png")


# ---------------------------------------------------------------------------
# Part 6 — abstention
# ---------------------------------------------------------------------------


def run_abstention_demo(llm, retriever, chunks) -> None:
    print()
    print("=" * 74)
    print("PART 6 — Does it admit when the answer is not in the documents?")
    print("=" * 74)

    def abstained(text: str) -> bool:
        t = text.lower()
        return any(p in t for p in ("not in context", "does not contain", "doesn't contain",
                                    "not mentioned", "no information", "not specified",
                                    "not provided", "cannot answer", "does not provide",
                                    "not stated", "no mention"))

    def run(questions, strict):
        out = []
        for item in questions:
            q = item if isinstance(item, str) else item[0]
            idx = retriever.retrieve(q, k=3)
            out.append(llm.ask(build_prompt(chunks, idx, q, strict=strict), max_tokens=120))
        return out

    # --- Easy: the topic is simply absent from the handbook.
    print("\nEASY CASE — the topic is entirely absent from the handbook.\n")
    print(f"{'question':<46}{'plain prompt':>15}{'strict prompt':>15}")
    print("-" * 74)
    plain = run(UNANSWERABLE, strict=False)
    strict = run(UNANSWERABLE, strict=True)
    p_ok = [abstained(a) for a in plain]
    s_ok = [abstained(a) for a in strict]
    for q, a, b in zip(UNANSWERABLE, p_ok, s_ok):
        print(f"{q[:44]:<46}{'abstained' if a else 'INVENTED':>15}"
              f"{'abstained' if b else 'INVENTED':>15}")
    print(f"\n{'abstention rate':<46}{np.mean(p_ok):>15.3f}{np.mean(s_ok):>15.3f}")

    print(
        f"\nBoth score {np.mean(p_ok):.1%}, and that is not the result the usual advice predicts.\n"
        f"The received wisdom is that you must tell the model to say 'I don't know' or it\n"
        f"will fabricate. Measured here, this model already refuses without being asked,\n"
        f"and the extra instruction changes nothing — because when a topic is completely\n"
        f"missing from the context, noticing that is easy."
    )

    # --- Hard: a plausible decoy number IS in the context.
    print("\n" + "-" * 74)
    print("HARD CASE — also unanswerable, but a plausible WRONG number is right there")
    print("in the retrieved context, waiting to be mistaken for the answer.")
    print("-" * 74 + "\n")
    print(f"{'question':<44}{'decoy':>8}{'plain':>11}{'strict':>11}")
    print("-" * 74)
    t_plain = run(TRAPS, strict=False)
    t_strict = run(TRAPS, strict=True)
    tp_ok, ts_ok = [], []
    fabrications = []
    for (q, decoy), a, b in zip(TRAPS, t_plain, t_strict):
        # Correct behaviour = abstain. Failure = report the decoy as the answer.
        fell_a = graded(a, decoy) and not abstained(a)
        fell_b = graded(b, decoy) and not abstained(b)
        tp_ok.append(not fell_a)
        ts_ok.append(not fell_b)
        if fell_a:
            fabrications.append((q, decoy, a))
        print(f"{q[:42]:<44}{decoy:>8}{'TRAPPED' if fell_a else 'ok':>11}"
              f"{'TRAPPED' if fell_b else 'ok':>11}")
    print(f"\n{'avoided the decoy':<44}{'':>8}{np.mean(tp_ok):>11.3f}{np.mean(ts_ok):>11.3f}")

    # --- The cost: does the strict instruction damage real answers?
    kept_plain, kept_strict = [], []
    for q, expected in QUESTIONS:
        idx = retriever.retrieve(q, k=3)
        kept_plain.append(graded(llm.ask(build_prompt(chunks, idx, q), max_tokens=120), expected))
        kept_strict.append(graded(llm.ask(build_prompt(chunks, idx, q, strict=True),
                                          max_tokens=120), expected))
    print(f"{'accuracy on ANSWERABLE questions':<44}{'':>8}"
          f"{np.mean(kept_plain):>11.3f}{np.mean(kept_strict):>11.3f}")

    delta = np.mean(kept_strict) - np.mean(kept_plain)
    print(
        f"\nThis is where the measured answer differs from the advice you will read\n"
        f"everywhere, so it is worth stating carefully rather than spinning:\n\n"
        f"  - The traps did NOT work either: {np.mean(tp_ok):.0%} avoided with a plain prompt. I built\n"
        f"    these specifically to fool the model — a hardware-budget question with '1,800'\n"
        f"    (the LEARNING budget) sitting in the context, a sick-leave question with '28'\n"
        f"    (ANNUAL leave) right there — and it declined every one. This model is more\n"
        f"    conservative than I expected.\n"
        f"  - The strict instruction therefore buys NOTHING here: {np.mean(tp_ok):.3f} -> {np.mean(ts_ok):.3f} on traps,\n"
        f"    {np.mean(p_ok):.3f} -> {np.mean(s_ok):.3f} on absent topics.\n"
        f"  - And it is not free: accuracy on genuinely answerable questions falls\n"
        f"    {np.mean(kept_plain):.3f} -> {np.mean(kept_strict):.3f} ({delta:+.3f}). Telling a model to refuse when unsure makes\n"
        f"    it refuse when it should not have.\n\n"
        f"On this corpus, with this model, the famous 'always tell it to say I do not know'\n"
        f"line is a NET NEGATIVE: no measurable benefit, a measurable cost. That is not a\n"
        f"claim that the advice is wrong in general — llama-3.1-8b-instant is heavily tuned\n"
        f"to hedge, a larger or less cautious model may well need the instruction, and\n"
        f"{len(UNANSWERABLE)} + {len(TRAPS)} questions is a small sample from which to prove a negative.\n\n"
        f"The transferable lesson is the one this curriculum keeps arriving at: prompt advice\n"
        f"is an empirical claim about YOUR model and YOUR data, and it takes about twenty\n"
        f"lines of code to check instead of adopting it. Project 16 builds that harness\n"
        f"properly.\n\n"
        f"Note also what none of this fixes. The model is following an instruction, not\n"
        f"reasoning about evidence. It has no way to know what it does not know."
    )

    if fabrications:
        q, decoy, a = fabrications[0]
        print(f"\nA trap being sprung — the decoy '{decoy}' belongs to a different policy:\n")
        print(f"  Q: {q}")
        print(f"  A: {a[:200]}")

    plt.figure(figsize=(8, 4.4))
    x = np.arange(3)
    plt.bar(x - 0.18, [np.mean(p_ok), np.mean(tp_ok), np.mean(kept_plain)], 0.36,
            label="plain prompt", color="steelblue")
    plt.bar(x + 0.18, [np.mean(s_ok), np.mean(ts_ok), np.mean(kept_strict)], 0.36,
            label="strict prompt", color="seagreen")
    plt.xticks(x, ["abstains when\ntopic absent", "avoids decoy\nin context",
                   "answers real\nquestions"])
    plt.ylabel("rate (higher is better in all three)")
    plt.ylim(0, 1.15)
    plt.title("The 'say I don't know' instruction is a trade, not a free win")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "abstention.png", dpi=120)
    plt.close()
    print("\nSaved plot to outputs/abstention.png")


# ---------------------------------------------------------------------------


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    for env_file in (HERE.parent / ".env", HERE / ".env"):
        if env_file.exists():
            try:
                from dotenv import load_dotenv

                load_dotenv(env_file)
            except ImportError:
                pass

    llm = LLM()
    if not llm.available and not any(CACHE_DIR.glob("*.json")):
        print("=" * 74)
        print("No GROQ_API_KEY found and no cached responses — generation parts will be")
        print("skipped. Retrieval (Parts 2 and 5's recall column) still works offline.")
        print("Put GROQ_API_KEY in ../.env to run the full project.")
        print("=" * 74)

    print("Loading the embedding model (local, ~90 MB on first run)...")
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    chunks = chunk_text(HANDBOOK)
    retriever = Retriever(chunks, embedder)
    print(f"Knowledge base: {len(HANDBOOK.split())} words -> {len(chunks)} chunks\n")

    run_problem_demo(llm, retriever, chunks)
    run_pipeline_demo(llm, retriever, chunks)
    run_retrieval_quality_demo(llm, retriever, chunks)
    run_topk_demo(llm, retriever, chunks)
    run_hybrid_demo(llm, retriever, chunks)
    run_abstention_demo(llm, retriever, chunks)

    print(f"\n{'=' * 74}")
    print(f"LLM calls made: {llm.calls} new, {llm.cached} served from cache.")
    print(f"Cached responses live in ./cache — delete it to force fresh calls.")


if __name__ == "__main__":
    main()
