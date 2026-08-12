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
- **03 — Model Evaluation & Validation** — next. train/val/test, k-fold CV,
  bias-variance, confusion matrix, precision/recall/F1/ROC-AUC.

## Repo

Remote: https://github.com/jyotisubham1/ML-GenAI (branch `main`, GPL-3.0).
Commit and push each project as it lands, alongside the README/Progress updates above.
The `.venv` directories are gitignored and must never be committed — they are ~365 MB
each and contain machine-specific absolute paths.

## Environment gotchas

- Each project has its own `.venv` (python 3.11) with a `requirements.txt`.
- **Avoid `&` and spaces in the folder name.** It was once `mal&genai`; renaming to
  `ML-GenAI` broke every venv console-script shebang (`bad interpreter`) because the
  old absolute path is baked into them. If a rename happens again, either recreate the
  venvs from `requirements.txt` or rewrite the stale path inside `.venv/bin/*` and
  `.venv/pyvenv.cfg`.
