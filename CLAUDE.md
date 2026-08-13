# Working agreement for this repo

This file travels with the folder, so it survives renames and fresh sessions. Read it
before doing anything else here, then read `README.md` for current progress.

## What this is

A learn-by-doing curriculum taking the user from classical ML → deep learning →
GenAI/RAG/LangChain/agents/evaluation, with the goal of becoming a high-level ML
engineer / data scientist / AI engineer. 18 projects, listed in `README.md`.

## How to teach here (non-negotiable)

The user's premise is: *"I believe in learn by doing — why is this done, how is it
working, why."* Every project must deliver, in this order:

1. **Intuition** in plain language — what problem is this solving, and why.
2. **The mathematical formula**, plus a derivation of *why it has that shape*. Not
   just "here is the formula" — show where it comes from.
3. **Formula → code mapping**, line by line, so the symbols in the math are visibly
   the variables in the code.
4. **The data**: where it comes from, and why it fits the concept being taught.
   Prefer a real dataset over a toy one whenever a real one exists.
5. **Training and evaluation**: how it's trained, what the metrics mean, what counts
   as good.
6. **Exercises** that break things on purpose — that is where intuition is built.

**Prove claims, don't assert them.** If a claim is testable, run the experiment and
show numbers and a plot. Example from project 02: rather than stating "MSE is the
wrong loss for classification," train a cross-entropy model and an MSE model from an
identical bad initialization and plot MSE's loss sitting flat at 0.987 while
cross-entropy collapses 24 → 0.04. A "because the theory says so" that *could* have
been demonstrated is a failure of this repo's whole premise.

## Cadence

Build **one project at a time, in order**, and only when the user says "next". Do not
build ahead. Answer questions about the current project freely.

## After finishing each project

1. Update the roadmap table in `README.md` so it links the new project.
2. Update this file's "Progress" section below.
3. Write/refresh a memory file (see `.claude` memory for this project) — but treat
   `README.md` and this file as the source of truth for status, since the memory
   directory is keyed by absolute path and is silently orphaned by a rename.

## Progress

- **01 — Linear Regression from Scratch** — done. MSE, hand-derived gradient descent,
  normal equation vs. iterative GD.
- **02 — Logistic Regression / Classification** — done. Sigmoid derived from the
  log-odds link, BCE derived from maximum likelihood, gradient derived by hand,
  breast-cancer dataset scratch-vs-sklearn, empirical BCE-vs-MSE demo.
- **03 — Model Evaluation & Validation** — done. Metrics from scratch, accuracy
  paradox on 3%-prevalence data, split variance vs. k-fold, bias-variance
  decomposition derived *and* verified numerically, ROC-AUC vs. average precision
  with a Monte-Carlo proof of the ranking interpretation, and a leakage demo hitting
  89% on pure noise.
- **04 — Trees & Ensembles** — done. Scratch tree on entropy/information gain, depth
  overfitting sweep, an out-of-sample test of the bagging variance formula, random
  forests measured to raise bias while cutting variance (and *losing* to plain bagging
  on this data — reported honestly), scratch gradient boosting, and a bias-variance
  table showing bagging fixes variance while boosting fixes bias.
- **05 — Clustering & Dimensionality Reduction** — done. Scratch k-means with its
  monotonic-decrease proof, k-means++ vs. random init over 200 restarts, elbow +
  silhouette, the two structural failures of k-means, PCA derived via Lagrange
  multipliers, and PCA beating 500 random projections.

- **06 — Neural Network from Scratch** — done. Forward pass, backprop derived via the
  chain rule and verified against numerical gradients to ~1e-9, XOR, the linear-collapse
  proof, vanishing gradients measured across 12 layers, and initialization tested at two
  depths. Reports honestly that logistic regression beats the network on digits.

- **07 — Neural Networks in PyTorch** — done. Autograd proven identical to project
  06's hand-derived backprop (6.9e-17), computation graphs with a hand-checked
  derivative, optimizer comparison on a stretched quadratic and a real network,
  mini-batch vs full-batch, a regularization *strength* sweep (over-regularization
  drops accuracy to chance), and a validation-selected pipeline that finally beats
  logistic regression on the same split.

- **08 — CNN Image Classifier** — done. Convolution from scratch vs F.conv2d, where a
  CNN's parameters really sit (the FC head holds 95%), translation robustness measured
  across three models, MNIST accuracy, data efficiency over 3 seeds, learned kernels
  matched against Sobel. Key finding: global average pooling is far MORE shift-robust
  (0.615 vs 0.112 at 4px) while being less accurate on centred data.
- **09 — RNN/LSTM Sequence Modeling** — done. RNN cell from scratch, gradient flow
  measured per timestep, memory task across four architectures, exploding gradients and
  clipping, LSTM gates inspected, char-level LM. Key finding: an UNTRAINED LSTM vanishes
  as badly as an RNN (the forget-gate bias is the actual fix, worth 12 orders of
  magnitude) — and the default LSTM fails a task the fb=3 version solves perfectly.
- **10 — Transformer from Scratch** — done. Attention from scratch vs PyTorch, sqrt(d_k)
  justified by measuring softmax entropy collapse, permutation equivariance proved,
  positional encoding shown necessary (0.495 vs 0.999), causal masking verified exact,
  mini-GPT. Payoff: transformer scores 1.000 on project 09's 80-step memory task where
  every recurrent model was at chance.

**Phases 1 and 2 complete.** Next is **11 — LLM Fundamentals: Tokenization &
Sampling** (BPE, temperature/top-k/top-p), which begins Phase 3 and switches from
building models to using pretrained ones.

### README style (set at 03, extended at 04 — apply to ALL projects)

Project READMEs are written for a beginner who has never seen the topic before.
Required elements, in order:

1. **§1 What you'll build** — with a table of claims and the evidence for each.
2. **§2 "What is X, why do we need it, where is it used?"** — plain language, BEFORE
   any mathematics. Must cover: what the thing actually is (concrete, ideally a
   picture or diagram); what's wrong with the previous project's method that motivates
   it; and real-world applications, including **when NOT to use it**.
3. **§3 The core idea** — the intuition that the math will formalize.
4. **§4 The math** — every formula gets **all three** of:
   - **"Reading it aloud"** — the equation spoken as an English sentence.
   - **A symbol table** — every symbol, how it's pronounced, and what it means *here*.
     Never assume Σ, Π, ∈, E[·], ρ, σ, ∂ or subscript/superscript notation is known.
   - **"Where it comes from"** — derived, defined, or assumed; and why it has that
     shape rather than another.
5. **§ Results** — generated plots **embedded inline**, each with a paragraph on what
   to look at, plus worked numeric examples using the script's real printed output.
6. **Exercises** that break things on purpose.

Math is **LaTeX** in `$…$` / `$$…$$` (GitHub renders it natively — never fall back to
plain-text formula blocks). Prefer more explanation over less: assume the reader is
smart but has forgotten all notation. Projects 01–03 were retrofitted to this style on
2026-08-12; keep all projects consistent.

## Repo

Remote: https://github.com/jyotisubham1/ML-GenAI (branch `main`, GPL-3.0).
Commit and push each project as it lands, alongside the README/Progress updates above.
The `.venv` directories are gitignored and must never be committed — they are ~365 MB
each and contain machine-specific absolute paths.

## LLM access for Phase 3 (projects 11+)

A **Groq** API key lives in `.env` as `GROQ_API_KEY` (gitignored at `.gitignore:151`,
never committed — verify before touching it). Groq is OpenAI-compatible:
`https://api.groq.com/openai/v1`, so the `openai` SDK works with `base_url` set.

Verified 2026-08-13: key valid, 15 models, including `llama-3.3-70b-versatile`,
`llama-3.1-8b-instant`, `openai/gpt-oss-120b`, `whisper-large-v3`.

**Groq has NO embeddings endpoint.** Use `sentence-transformers` locally
(`all-MiniLM-L6-v2`) for project 12 and the retrieval half of 13/14 — free, offline,
and better for teaching since the vectors are inspectable.

Which projects actually need the API:
- **11 (tokenization/sampling): NO.** Sampling operates on the full logit vector, which
  a hosted API does not expose. Use project 10's mini-GPT and/or a local GPT-2.
- **12 (embeddings): NO** — local embedding model.
- **13, 14, 15, 16: YES** — they need real generation.
- **17 (fine-tuning): mostly local**, since Groq serves models but does not train them.

Every project must degrade gracefully without a key: if `GROQ_API_KEY` is missing, skip
the generation parts with a clear message rather than crashing, so the repo stays
runnable for anyone who clones it.

## Environment gotchas

- Each project has its own `.venv` (python 3.11) with a `requirements.txt`.
- **Avoid `&` and spaces in the folder name.** It was once `mal&genai`; renaming to
  `ML-GenAI` broke every venv console-script shebang (`bad interpreter`) because the
  old absolute path is baked into them. If a rename happens again, either recreate the
  venvs from `requirements.txt` or rewrite the stale path inside `.venv/bin/*` and
  `.venv/pyvenv.cfg`.
