"""
RAG with LangChain — project 13's pipeline rebuilt on a framework, then compared
against it directly. Six experiments:

  Part 1  the same four stages, in LangChain components
  Part 2  verification: does the framework produce IDENTICAL results to the raw code?
  Part 3  what LCEL actually buys — batching and async, measured in seconds
  Part 4  swappability: changing the vector store and the retriever in one line
  Part 5  the cost of the abstraction, measured (dependencies, install size, imports)
  Part 6  where the abstraction leaks, and what to do about it

Imports project 13's handbook and questions directly, so the comparison uses the
same data, the same model and the same prompt text. Any difference in results is
then attributable to the framework, not to the setup.

Run:
    python rag_langchain.py
"""

import importlib.util
import os
import subprocess
import sys
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
SCRATCH_DIR = HERE.parent / "13-rag-from-scratch"
MODEL = "llama-3.1-8b-instant"


def load_project_13():
    """
    Import project 13's module so this project uses the IDENTICAL handbook,
    questions, chunking and prompt text. Comparing two pipelines is only
    meaningful if everything except the framework is held constant.
    """
    spec = importlib.util.spec_from_file_location(
        "scratch_rag", SCRATCH_DIR / "rag_from_scratch.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["scratch_rag"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Part 1 — the same pipeline, in framework components
# ---------------------------------------------------------------------------


def build_langchain_pipeline(scratch):
    """
    Project 13's four stages, each replaced by a LangChain component:

        chunk     -> RecursiveCharacterTextSplitter
        embed     -> HuggingFaceEmbeddings (the same MiniLM model, locally)
        retrieve  -> FAISS vector store .as_retriever()
        augment   -> ChatPromptTemplate
        generate  -> ChatGroq
        wiring    -> LCEL, the | operator                                      (1)
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_groq import ChatGroq
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=350, chunk_overlap=120, separators=["\n\n", "\n", ". ", " "])
    docs = splitter.create_documents([scratch.HANDBOOK])

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    store = FAISS.from_documents(docs, embeddings)
    retriever = store.as_retriever(search_kwargs={"k": 3})

    # The prompt text is copied from project 13 verbatim, so any behavioural
    # difference cannot be blamed on wording.
    prompt = ChatPromptTemplate.from_template(scratch.PROMPT)
    llm = ChatGroq(model=MODEL, temperature=0, max_tokens=120)

    def format_docs(retrieved):
        return "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(retrieved))

    # (1) LCEL: each | passes the previous output into the next component.
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever, store, docs


def run_pipeline_demo(scratch, chain, docs) -> None:
    print("=" * 74)
    print("PART 1 — The same four stages, now as framework components")
    print("=" * 74)

    print(f"\n{'stage':<12}{'project 13 (raw)':<34}{'project 14 (LangChain)':<28}")
    print("-" * 74)
    for stage, raw, lc in (
        ("chunk", "chunk_text(size, overlap)", "RecursiveCharacterTextSplitter"),
        ("embed", "SentenceTransformer.encode()", "HuggingFaceEmbeddings"),
        ("store", "a numpy array", "FAISS vector store"),
        ("retrieve", "vecs @ q, argsort", "store.as_retriever()"),
        ("augment", "PROMPT.format(...)", "ChatPromptTemplate"),
        ("generate", "openai client + cache", "ChatGroq"),
        ("wiring", "four function calls", "LCEL: a | b | c"),
    ):
        print(f"{stage:<12}{raw:<34}{lc:<28}")

    print(f"\nSame handbook ({len(scratch.HANDBOOK.split())} words), now {len(docs)} chunks "
          f"(project 13 made {len(scratch.chunk_text(scratch.HANDBOOK))}).")

    question = "How many days of leave can be carried over, and when do they expire?"
    print(f"\nQuestion: \"{question}\"")
    answer = chain.invoke(question)
    print(f"Answer:   {answer[:180]}")

    print(
        "\nThe whole chain is one expression:\n\n"
        "    chain = {\"context\": retriever | format_docs,\n"
        "             \"question\": RunnablePassthrough()} | prompt | llm | StrOutputParser()\n\n"
        "Read the | as 'feed into'. That is genuinely more readable than four function\n"
        "calls threaded together by hand — and it is also the point at which you stop\n"
        "seeing the retrieved text unless you go looking for it. Part 6 is about that."
    )


# ---------------------------------------------------------------------------
# Part 2 — does the framework change the answers?
# ---------------------------------------------------------------------------


def run_equivalence_demo(scratch, chain, retriever) -> None:
    print()
    print("=" * 74)
    print("PART 2 — Verification: same data, same model — are the results identical?")
    print("=" * 74)

    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    chunks = scratch.chunk_text(scratch.HANDBOOK)
    raw_retriever = scratch.Retriever(chunks, embedder)
    llm = scratch.LLM()

    print(f"\n{'question':<40}{'raw':>9}{'LangChain':>12}{'agree?':>9}")
    print("-" * 74)
    raw_ok, lc_ok, agree = [], [], []
    for q, expected in scratch.QUESTIONS:
        idx = raw_retriever.retrieve(q, k=3)
        raw_answer = llm.ask(scratch.build_prompt(chunks, idx, q), max_tokens=120)
        lc_answer = chain.invoke(q)
        r, l = scratch.graded(raw_answer, expected), scratch.graded(lc_answer, expected)
        raw_ok.append(r)
        lc_ok.append(l)
        agree.append(r == l)
        print(f"{q[:38]:<40}{'YES' if r else 'no':>9}{'YES' if l else 'no':>12}"
              f"{'yes' if r == l else 'NO':>9}")

    print(f"\n{'accuracy':<40}{np.mean(raw_ok):>9.3f}{np.mean(lc_ok):>12.3f}"
          f"{np.mean(agree):>9.3f}")

    # Where they disagree, diagnose it with project 13's own rule: was the answer
    # even retrieved? That separates a retrieval failure from a generation failure.
    disagreements = []
    for (q, expected), r, l in zip(scratch.QUESTIONS, raw_ok, lc_ok):
        if r != l:
            lc_context = " ".join(d.page_content for d in retriever.invoke(q))
            raw_context = " ".join(
                chunks[i] for i in raw_retriever.retrieve(q, k=3))
            disagreements.append((q, expected,
                                  scratch.graded(raw_context, expected),
                                  scratch.graded(lc_context, expected),
                                  llm.ask(scratch.build_prompt(
                                      chunks, raw_retriever.retrieve(q, k=3), q),
                                      max_tokens=120),
                                  chain.invoke(q)))

    print(
        f"\nNot identical: raw {np.mean(raw_ok):.3f}, LangChain {np.mean(lc_ok):.3f}, agreeing on "
        f"{np.mean(agree):.0%} of questions.\n\n"
        f"I expected these to match exactly, and it is worth being clear about why they do\n"
        f"not. Everything that could change the ANSWER is held constant — same embedding\n"
        f"model, same similarity metric, same prompt text, same model at temperature 0.\n"
        f"What differs is CHUNKING: project 13 splits on word counts, while\n"
        f"RecursiveCharacterTextSplitter splits on paragraph and sentence boundaries.\n"
        f"Different chunks means different retrieval, which means a different context."
    )

    for q, expected, raw_had, lc_had, raw_ans, lc_ans in disagreements:
        print(
            f"\nThe question they disagree on:\n"
            f"  Q: {q}   (expected: '{expected}')\n"
            f"  answer present in raw context?        {'YES' if raw_had else 'NO'}\n"
            f"  answer present in LangChain context?  {'YES' if lc_had else 'NO'}"
        )
        if raw_had and not lc_had:
            print(
                "\n  So this is a RETRIEVAL failure, not a generation failure — exactly the\n"
                "  distinction project 13's Part 3 said to check first. The smarter splitter\n"
                "  produced a chunk boundary that separated this fact from the words needed\n"
                "  to find it.\n\n"
                "  Worth sitting with: RecursiveCharacterTextSplitter is the better tool by\n"
                "  reputation, and on this question it retrieved worse. Framework defaults are\n"
                "  defaults, not answers, and project 12's Part 6 already showed chunking\n"
                "  swinging accuracy from 0.222 to 1.000. Nothing about using a framework\n"
                "  removes the need to measure your own chunking."
            )
        elif lc_had and not raw_had:
            print("\n  Here the framework's chunking retrieved BETTER than the hand-rolled one.")
        else:
            print(
                f"\n  Both contexts contain the answer, so this is a GENERATION difference.\n\n"
                f"  raw:       {raw_ans[:110]}\n"
                f"  LangChain: {lc_ans[:110]}\n\n"
                f"  Do not read this as 'temperature 0 is unreliable'. The model is\n"
                f"  deterministic; the PROMPTS were not identical. Same template, but the\n"
                f"  different chunk boundaries produced a different context string, and a\n"
                f"  different input legitimately gives a different output.\n\n"
                f"  The lesson is about the size of the lever: a change to chunking that left\n"
                f"  recall untouched — the answer was retrieved either way — still changed the\n"
                f"  final answer. Retrieval recall is necessary but not sufficient; how the\n"
                f"  surrounding text frames the fact matters too."
            )


# ---------------------------------------------------------------------------
# Part 3 — what LCEL buys
# ---------------------------------------------------------------------------


def run_lcel_demo(scratch, chain) -> None:
    print()
    print("=" * 74)
    print("PART 3 — What LCEL actually buys: batch and async, measured")
    print("=" * 74)

    questions = [q for q, _ in scratch.QUESTIONS]

    t0 = time.perf_counter()
    for q in questions:
        chain.invoke(q)
    sequential = time.perf_counter() - t0

    t0 = time.perf_counter()
    chain.batch(questions)
    batched = time.perf_counter() - t0

    print(f"\n{len(questions)} questions through the same chain:\n")
    print(f"{'method':<34}{'seconds':>12}{'speedup':>11}")
    print("-" * 74)
    print(f"{'chain.invoke() in a loop':<34}{sequential:>12.2f}{1.0:>10.1f}x")
    print(f"{'chain.batch() — parallel':<34}{batched:>12.2f}{sequential / batched:>10.1f}x")

    print(
        f"\nOne method call, {sequential / batched:.1f}x faster. This is the strongest practical argument for\n"
        f"the framework: .batch(), .stream(), .astream() and .ainvoke() come free on every\n"
        f"chain you build, because every LCEL component implements the same Runnable\n"
        f"interface. Writing the async and batching logic yourself in project 13 would have\n"
        f"been real work — thread pools, error handling per item, ordering guarantees.\n\n"
        f"Streaming is the same story. chain.stream(q) yields tokens as they arrive, which\n"
        f"is what makes a chat UI feel responsive, and it required no change to the chain."
    )

    # Streaming, shown rather than described.
    print(f"\nStreaming the first tokens of an answer as they arrive:\n  ", end="", flush=True)
    got = []
    for piece in chain.stream(questions[0]):
        got.append(piece)
        if len("".join(got)) > 90:
            break
    print("".join(got).replace("\n", " ") + " ...")
    print(f"  ({len(got)} chunks streamed rather than one blocking response)")

    plt.figure(figsize=(7, 4))
    plt.bar(["invoke() in a loop", "batch()"], [sequential, batched],
            color=["indianred", "seagreen"])
    for i, v in enumerate([sequential, batched]):
        plt.text(i, v + 0.1, f"{v:.1f}s", ha="center")
    plt.ylabel("seconds for 8 questions")
    plt.title(f"LCEL gives parallelism for free ({sequential / batched:.1f}x here)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lcel_batching.png", dpi=120)
    plt.close()
    print("\nSaved plot to outputs/lcel_batching.png")


# ---------------------------------------------------------------------------
# Part 4 — swappability
# ---------------------------------------------------------------------------


def run_swappability_demo(scratch, docs) -> None:
    print()
    print("=" * 74)
    print("PART 4 — Swapping components: the real selling point")
    print("=" * 74)

    from langchain_community.vectorstores import FAISS
    from langchain_community.retrievers import BM25Retriever

    # EnsembleRetriever has moved between packages more than once. In LangChain 1.x
    # it lives in `langchain_classic` — a package name that makes Part 5's point
    # about API churn better than any paragraph could. This tolerant import is what
    # you end up writing in real projects.
    try:
        from langchain_classic.retrievers import EnsembleRetriever
    except ImportError:  # older layouts
        from langchain.retrievers import EnsembleRetriever
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    dense = FAISS.from_documents(docs, embeddings).as_retriever(search_kwargs={"k": 3})
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 3
    hybrid = EnsembleRetriever(retrievers=[dense, bm25], weights=[0.5, 0.5])

    print("\nThree retrievers, each one line to construct. Measuring recall@3 —")
    print("does the retrieved context contain the answer string at all?\n")
    print(f"{'retriever':<38}{'recall@3':>12}")
    print("-" * 74)
    results = {}
    for name, r in (("FAISS dense", dense), ("BM25 keyword", bm25),
                    ("EnsembleRetriever (hybrid)", hybrid)):
        hits = []
        for q, expected in scratch.QUESTIONS:
            context = " ".join(d.page_content for d in r.invoke(q))
            hits.append(scratch.graded(context, expected))
        results[name] = float(np.mean(hits))
        print(f"{name:<38}{results[name]:>12.3f}")

    print(
        "\nProject 13 implemented reciprocal rank fusion by hand — about 8 lines. Here it is\n"
        "EnsembleRetriever(retrievers=[dense, bm25]), and it uses RRF internally with the\n"
        "same K=60 constant. That is a fair trade when you want the standard thing.\n\n"
        "The deeper point is the interface. Every retriever exposes .invoke(query) -> list\n"
        "of Documents, so swapping FAISS for Chroma, Pinecone or Weaviate changes one line\n"
        "and nothing downstream. Same for the model: ChatGroq -> ChatOpenAI -> ChatAnthropic.\n"
        "In project 13, changing the vector store would have meant rewriting the retrieval\n"
        "function, and changing the provider would have meant rewriting the client.\n\n"
        "This is what frameworks are actually for. Not making the first version easier —\n"
        "project 13 was already only 40 lines — but making the tenth CHANGE easier."
    )

    plt.figure(figsize=(7.5, 4.2))
    plt.bar(list(results), list(results.values()),
            color=["steelblue", "darkorange", "seagreen"])
    plt.ylabel("recall@3")
    plt.ylim(0, 1.15)
    plt.xticks(rotation=8, fontsize=8)
    plt.title("Three retrievers, one line each")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "retrievers.png", dpi=120)
    plt.close()
    print("Saved plot to outputs/retrievers.png")


# ---------------------------------------------------------------------------
# Part 5 — the cost
# ---------------------------------------------------------------------------


def run_cost_demo() -> None:
    print()
    print("=" * 74)
    print("PART 5 — The cost of the abstraction, measured")
    print("=" * 74)

    def venv_stats(project_dir: Path):
        site = list((project_dir / ".venv" / "lib").glob("python*/site-packages"))
        if not site:
            return None, None
        packages = len([p for p in site[0].iterdir()
                        if p.is_dir() and not p.name.startswith("_")
                        and p.name.endswith(".dist-info")])
        size = subprocess.run(["du", "-sm", str(project_dir / ".venv")],
                              capture_output=True, text=True).stdout.split()[0]
        return packages, int(size)

    print(f"\n{'project':<34}{'installed packages':>20}{'venv size (MB)':>17}")
    print("-" * 74)
    stats = {}
    for label, path in (("13 — raw (openai + st)", SCRATCH_DIR),
                        ("14 — LangChain", HERE)):
        pkgs, size = venv_stats(path)
        stats[label] = (pkgs, size)
        if pkgs:
            print(f"{label:<34}{pkgs:>20}{size:>17}")

    # Import time is paid on every cold start — serverless, CLI tools, tests.
    # Import exactly what each project needs to serve one query, so the comparison
    # is fair. Both need sentence_transformers (and therefore torch) for the local
    # embedding model — an earlier version of this measurement omitted it from the
    # LangChain side and made the framework look FASTER to import, which was an
    # artefact of the benchmark rather than a property of the framework.
    code_raw = "import openai, sentence_transformers"
    code_lc = ("import sentence_transformers, langchain, langchain_core, "
               "langchain_community, langchain_groq, langchain_huggingface, "
               "langchain_text_splitters")
    times = {}
    for label, code, python in (("raw imports", code_raw, SCRATCH_DIR / ".venv/bin/python"),
                                ("LangChain imports", code_lc, HERE / ".venv/bin/python")):
        t0 = time.perf_counter()
        subprocess.run([str(python), "-c", code], capture_output=True)
        times[label] = time.perf_counter() - t0
    print(f"\n{'cold import time':<34}{'':>20}{'seconds':>17}")
    print("-" * 74)
    for label, t in times.items():
        print(f"{label:<34}{'':>20}{t:>17.2f}")

    raw_pkgs = stats["13 — raw (openai + st)"][0]
    lc_pkgs = stats["14 — LangChain"][0]
    raw_size = stats["13 — raw (openai + st)"][1]
    lc_size = stats["14 — LangChain"][1]
    t_raw = times["raw imports"]
    t_lc = times["LangChain imports"]
    print(
        f"\nLangChain pulls in {lc_pkgs} packages against {raw_pkgs}"
        f"{f' ({lc_pkgs / raw_pkgs:.1f}x)' if raw_pkgs else ''}, adds {lc_size - raw_size} MB to the virtualenv\n"
        f"({lc_size / raw_size:.2f}x), and costs {t_lc - t_raw:+.2f}s of cold import time "
        f"({t_lc / t_raw:.2f}x).\n\n"
        f"THAT IS SMALLER THAN THE FRAMEWORK'S REPUTATION SUGGESTS, and it is worth saying\n"
        f"so rather than repeating the usual complaint. The reason is that both projects\n"
        f"already depend on sentence-transformers, which drags in PyTorch — roughly a\n"
        f"gigabyte on its own. Against that, LangChain's own footprint is a rounding error.\n"
        f"If you were calling a hosted embedding API instead of running MiniLM locally, the\n"
        f"ratio would look far worse for the framework.\n\n"
        f"The other cost does not show up in these numbers: API churn. LangChain has\n"
        f"reorganised its packages repeatedly (langchain -> langchain_community ->\n"
        f"langchain_huggingface, AgentExecutor -> LangGraph), so tutorials written eighteen\n"
        f"months ago frequently do not run. Project 13's forty lines of numpy and one HTTP\n"
        f"call will still work in five years.\n\n"
        f"None of this is an argument against the framework. It is an argument for choosing\n"
        f"deliberately: use it when you need the interchangeability of Part 4 and the free\n"
        f"batching of Part 3, and skip it when you need four function calls that never change."
    )

    plt.figure(figsize=(7.5, 4.2))
    labels = list(stats)
    x = np.arange(len(labels))
    plt.bar(x, [stats[l][1] for l in labels], color=["seagreen", "indianred"])
    for i, l in enumerate(labels):
        plt.text(i, stats[l][1] + 20, f"{stats[l][1]} MB\n{stats[l][0]} packages", ha="center",
                 fontsize=8)
    plt.xticks(x, ["raw (project 13)", "LangChain (project 14)"])
    plt.ylabel("virtualenv size (MB)")
    plt.title("What the convenience weighs")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dependency_cost.png", dpi=120)
    plt.close()
    print("Saved plot to outputs/dependency_cost.png")


# ---------------------------------------------------------------------------
# Part 6 — where it leaks
# ---------------------------------------------------------------------------


def run_debugging_demo(scratch, chain, retriever) -> None:
    print()
    print("=" * 74)
    print("PART 6 — Where the abstraction leaks: debugging a retrieval failure")
    print("=" * 74)

    print("\nProject 13's Part 3 established the rule: when RAG gives a bad answer, the")
    print("cause is usually retrieval, not generation. So the question that matters for")
    print("any framework is — how quickly can you see what was actually retrieved?\n")

    question = "What is the bonus for resolving a SEV1 incident?"  # a trap: no such bonus
    answer = chain.invoke(question)
    print(f"Question: \"{question}\"")
    print(f"Answer:   {answer[:150]}")

    print(f"\nThe chain gave you a string. To find out WHY, you have to reach inside it:\n")
    retrieved = retriever.invoke(question)
    for i, d in enumerate(retrieved):
        print(f"  chunk {i}: \"{d.page_content[:64].strip()}...\"")

    print(
        "\nThat required calling the retriever separately — the chain itself does not hand\n"
        "back its intermediate values. Project 13 printed the cosine score of every chunk\n"
        "as a matter of course, because there was nothing hiding it.\n\n"
        "LangChain's answers to this are real, and worth knowing:\n"
        "  - build the chain with RunnableParallel so it returns context alongside the\n"
        "    answer, instead of only the answer;\n"
        "  - set_debug(True) or set_verbose(True) for a trace of every step;\n"
        "  - LangSmith, which records each run's inputs and outputs (a paid hosted service).\n\n"
        "The honest summary of this project: the framework is worth it when you need\n"
        "swappable components (Part 4) and free concurrency (Part 3), and it costs you\n"
        "dependency weight (Part 5) and directness (this part). Having now built the same\n"
        "pipeline both ways, you can make that call on evidence instead of on fashion —\n"
        "which is the entire reason project 13 came first."
    )


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    for env_file in (HERE.parent / ".env", HERE / ".env"):
        if env_file.exists():
            try:
                from dotenv import load_dotenv

                load_dotenv(env_file)
            except ImportError:
                pass

    if not os.environ.get("GROQ_API_KEY"):
        print("No GROQ_API_KEY found in ../.env — this project needs it for generation.")
        print("Retrieval parts (4) would still work; exiting for clarity.")
        return

    print("Loading project 13's handbook, questions and prompt for a like-for-like test...")
    scratch = load_project_13()
    chain, retriever, store, docs = build_langchain_pipeline(scratch)
    print(f"Ready: {len(docs)} chunks in a FAISS index.\n")

    run_pipeline_demo(scratch, chain, docs)
    run_equivalence_demo(scratch, chain, retriever)
    run_lcel_demo(scratch, chain)
    run_swappability_demo(scratch, docs)
    run_cost_demo()
    run_debugging_demo(scratch, chain, retriever)


if __name__ == "__main__":
    main()
