# ML → DL → GenAI Learning Path

A learn-by-doing curriculum: classical machine learning → deep learning → generative
AI / RAG / LangChain / agents / evaluation. Every project is self-contained, runnable,
and explains **why**, not just **how**:

- the intuition in plain language,
- the actual math formula and a short derivation of why it's shaped that way,
- a line-by-line mapping from formula → code,
- where the data comes from and why it fits the concept,
- how training/evaluation works and what the metrics mean,
- exercises to build intuition by breaking things on purpose.

No project assumes you've memorized the last one, but each does build on ideas from
earlier ones — go in order the first time through.

## How to use this repo

1. Read `_shared/setup.md` once to set up Python.
2. `cd` into a numbered project folder, read its `README.md` top to bottom, then run
   the code yourself before reading the solution/explanation sections again.
3. Do the exercises. This is the part that actually builds intuition — running someone
   else's working code teaches far less than breaking it and figuring out why it broke.
4. Move to the next project.

Projects are built one at a time, in order, as you work through them — **01 (Linear
Regression)**, **02 (Logistic Regression)** and **03 (Model Evaluation)** exist so far.
Say "next" (or ask questions about the current one) when you're ready and the next one
gets built the same way: README + math + code + data + exercises.

## Roadmap

### Phase 1 — Classical ML foundations
| # | Project | Core ideas |
|---|---------|------------|
| 01 | [Linear Regression from Scratch](01-linear-regression-from-scratch/) | MSE loss, gradient descent derived by hand, normal equation vs. iterative GD |
| 02 | [Logistic Regression / Classification](02-logistic-regression-classification/) | sigmoid, cross-entropy, why MSE is wrong for classification, decision boundaries |
| 03 | [Model Evaluation & Validation](03-model-evaluation-validation/) | train/val/test, k-fold CV, bias-variance derived and verified numerically, confusion matrix, precision/recall/F1/ROC-AUC, data leakage |
| 04 | Trees & Ensembles | entropy/information gain, decision trees, random forest, gradient boosting |
| 05 | Clustering & Dimensionality Reduction | k-means objective, PCA via eigen-decomposition |

### Phase 2 — Deep learning foundations
| # | Project | Core ideas |
|---|---------|------------|
| 06 | Neural Network from Scratch (numpy) | forward pass, backprop derived via chain rule |
| 07 | Neural Network in PyTorch | autograd vs. your scratch backprop, `nn.Module`, optimizers |
| 08 | CNN Image Classifier | convolution math, kernels, stride/padding, pooling |
| 09 | RNN/LSTM Sequence Modeling | recurrence, vanishing/exploding gradients, LSTM gates |
| 10 | Transformer from Scratch | scaled dot-product attention derived, positional encoding, mini-GPT |

### Phase 3 — Generative AI / LLMs / RAG
| # | Project | Core ideas |
|---|---------|------------|
| 11 | LLM Fundamentals: Tokenization & Sampling | BPE tokenization, temperature/top-k/top-p math, prompting |
| 12 | Embeddings & Vector Search | cosine similarity derived, brute-force vs. FAISS index |
| 13 | RAG from Scratch (no framework) | chunking, retrieval, augmentation, generation |
| 14 | RAG with LangChain | same pipeline via LangChain retrievers/LCEL |
| 15 | Agents & Tool Use | ReAct pattern, LangChain tool-calling agents, failure modes |
| 16 | GenAI Evaluation & Metrics | precision@k/recall@k/MRR/NDCG, RAGAS-style faithfulness, LLM-as-judge |
| 17 | Fine-tuning & PEFT | LoRA math (low-rank weight updates), when to fine-tune vs. prompt |

### Capstone
| # | Project | Core ideas |
|---|---------|------------|
| 18 | Capstone RAG Assistant | combine retrieval + agents + evaluation into one end-to-end system |
