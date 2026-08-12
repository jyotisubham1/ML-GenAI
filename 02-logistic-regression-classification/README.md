# 02 — Logistic Regression / Classification

## 1. What you'll build

A binary classifier implemented with nothing but numpy — same gradient-descent
skeleton as project 01, but for predicting a *category* instead of a number. You'll
derive why the model needs a squashing function (sigmoid) and a different loss
function (cross-entropy) than linear regression used, prove the resulting gradient
has a surprisingly familiar shape, and then run an experiment that makes the "why not
just reuse MSE" question concrete instead of theoretical: you'll watch a model trained
with MSE get permanently stuck, and one trained with cross-entropy escape the same bad
starting point.

## 2. The core idea

Linear regression predicts an unbounded number. That's wrong for "is this tumor
malignant?" — the answer is 0 or 1, and anything a model outputs in between should be
read as a *probability*, not a raw score. Two things need to change from project 01:

1. **Squash the linear score into a probability** — a function that takes any real
   number and maps it to `(0, 1)`.
2. **Score "how good is this probability estimate" differently** — squared error
   treats being 40% confident and being 90% confident as a smooth, symmetric penalty;
   what we actually want is a loss that punishes *confident, wrong* predictions much
   more than MSE does, since a probability estimate that's confidently wrong is a much
   worse mistake than a hedged one.

## 3. The math

### The model and the sigmoid

```
z     = X @ w + b                          (the same linear score as before)
y_hat = sigmoid(z) = 1 / (1 + e^(-z))
```

Why *this* function, specifically? Two properties, and they're really the same
property stated two ways:

- Its range is exactly `(0, 1)` for any real input — a valid probability, no matter
  how extreme `z` gets.
- It's the inverse of the **log-odds** (logit) function: if `p = sigmoid(z)`, then
  `z = log(p / (1-p))`. So the linear model isn't predicting the probability directly —
  it's predicting the log-odds of the positive class, and the sigmoid converts that
  back to a probability. This is *why* the decision boundary ends up being a straight
  line (or flat hyperplane) even though the probability surface itself is curved: the
  boundary is where `p = 0.5`, which is exactly where `z = 0` — a linear condition on
  `x`. You can see this directly in `outputs/decision_boundary.png` after running the
  code: a straight line separates the two classes even though the underlying
  probabilities curve smoothly from 0 to 1 around it.

### The loss: binary cross-entropy, derived from maximum likelihood

Treat each label `y_i ∈ {0, 1}` as a draw from a Bernoulli distribution with success
probability `y_hat_i`. The probability of the data we actually observed, for one
point, is:

```
P(y_i | x_i) = y_hat_i^(y_i) * (1 - y_hat_i)^(1 - y_i)
```

(Check it: if `y_i=1` this is `y_hat_i`; if `y_i=0` this is `1 - y_hat_i` — exactly
the probability the model assigned to whichever outcome actually happened.)

The likelihood of the *whole* dataset is the product over all points (assuming
independence). Products of many small numbers underflow and are hard to differentiate,
so take the log instead — maximizing the log-likelihood is equivalent to maximizing
the likelihood, since log is monotonic:

```
log L(w,b) = Σ [ y_i * log(y_hat_i) + (1 - y_i) * log(1 - y_hat_i) ]
```

"Best" weights maximize this. Optimizers conventionally *minimize*, so negate and
average over `n` — that negative average log-likelihood is exactly the binary
cross-entropy loss:

```
J(w, b) = -(1/n) * Σ [ y*log(y_hat) + (1-y)*log(1-y_hat) ]
```

This isn't an arbitrary choice the way it might look — it's *the* loss implied by
"assume the labels are Bernoulli draws and find the weights that make the observed
data most probable." (Project 01's MSE has the same kind of justification: it's the
maximum-likelihood loss if you assume the errors are Gaussian — worth noticing that
"pick a loss" is usually really "pick an assumption about the noise.")

### The gradient — and why it looks exactly like project 01's

Differentiate `J` with respect to `z` (not `w` yet) using the chain rule through
`y_hat = sigmoid(z)`. Two facts you need first:

```
d(log y_hat)/dy_hat = 1/y_hat
sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z)) = y_hat * (1 - y_hat)
```

(That second one is a nice fact about the sigmoid worth deriving once on paper — it's
one of the few common activation functions whose derivative is this cheap to compute
from its own output, which is part of why it was historically popular.)

Now chain through `J -> y_hat -> z`:

```
dJ/dy_hat = -(1/n) * [ y/y_hat - (1-y)/(1-y_hat) ]

dJ/dz = dJ/dy_hat * dy_hat/dz
      = -(1/n) * [ y/y_hat - (1-y)/(1-y_hat) ] * y_hat*(1-y_hat)
      = -(1/n) * [ y*(1-y_hat) - (1-y)*y_hat ]              <- multiply through
      = -(1/n) * [ y - y*y_hat - y_hat + y*y_hat ]           <- expand
      = -(1/n) * [ y - y_hat ]
      = (1/n) * (y_hat - y)
```

The `y_hat*(1-y_hat)` sigmoid-derivative term **cancels exactly** against the
`1/y_hat` and `1/(1-y_hat)` terms from the cross-entropy derivative. That cancellation
is the whole point — it's not a coincidence, and it doesn't happen for other
loss/activation pairings (see Part 3 below, where using MSE instead breaks the
cancellation and the leftover `y_hat*(1-y_hat)` term causes real problems).

Stacked across features, the gradients are:

```
dJ/dw = (1/n) * X^T @ (y_hat - y)
dJ/db = (1/n) * Σ (y_hat - y)
```

**Compare to project 01's**: `dJ/dw = (2/n) * X^T @ (y_hat - y)` for linear
regression. Same `(y_hat - y)` error term, same `X^T @ (...)` structure — only the
constant differs (project 01's 2 comes from differentiating a square; here there's no
square). This isn't a coincidence either: linear regression (Gaussian noise + identity
link) and logistic regression (Bernoulli noise + logit link) are both members of the
same family of models (generalized linear models), and "gradient = error × input"
is the general pattern for that whole family.

## 4. From formula to code

Open [`logistic_regression.py`](logistic_regression.py) — the `LogisticRegressionGD`
class docstring numbers each formula, and the matching line in `fit()` carries the
same number.

| Formula | Code |
|---|---|
| `z = X @ w + b` | `z = X @ self.weights + self.bias` |
| `y_hat = sigmoid(z)` | `y_hat = self._sigmoid(z)` |
| `J = -(1/n)Σ[y·log(y_hat) + (1-y)·log(1-y_hat)]` | the `loss = -np.mean(...)` line |
| `dJ/dz = y_hat - y` | `dz = y_hat - y` |
| `dJ/dw = (1/n) X^T(y_hat-y)` | `dw = (1/n) * (X.T @ dz)` |
| `w := w - α·dJ/dw` | `self.weights -= self.learning_rate * dw` |

`predict_proba` returns `y_hat` directly (the actual probability estimate — useful
when you care about confidence, not just the label); `predict` thresholds it at 0.5
by default to get a hard 0/1 label.

## 5. The data

Three datasets/scenarios, each isolating a different idea:

1. **Synthetic 2D blobs** (`run_synthetic_demo`, Part 1): two well-separated Gaussian
   clusters. Because it's 2D, you can actually plot the learned decision boundary on
   top of the data and see the "linear boundary from a linear log-odds model" claim
   directly, not just algebraically.
2. **Breast cancer diagnosis** (`run_breast_cancer_demo`, Part 2): a real, built-in
   scikit-learn dataset — 30 features derived from cell measurements, predicting
   malignant vs. benign. Real features, real scale problems (another reason to
   standardize), and a real place where precision/recall tradeoffs actually matter
   (missing a malignant tumor and misclassifying a benign one are not equally bad —
   project 03 goes deep on this).
3. **The same 2D blobs, but from a bad starting point** (`run_loss_comparison_demo`,
   Part 3): both models start from identical, deliberately wrong initial weights
   (pointing away from the true separating direction) instead of the usual zero-init,
   to expose the difference between cross-entropy's and MSE's gradients when the model
   is confidently wrong — see below.

## 6. Build it

Everything's in [`logistic_regression.py`](logistic_regression.py):
- `LogisticRegressionGD` — from-scratch model; `loss="bce"` (default, correct) or
  `loss="mse"` (wrong on purpose, for Part 3's comparison).
- `run_synthetic_demo()` — 2D fit, loss curve, decision boundary plot.
- `run_breast_cancer_demo()` — scratch vs. scikit-learn on real data.
- `run_loss_comparison_demo()` — the bad-init BCE-vs-MSE experiment.

## 7. Train & evaluate

```bash
cd 02-logistic-regression-classification
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python logistic_regression.py
```

Expect:
- **Part 1**: ~98% test accuracy on the synthetic blobs (they're well-separated, so
  this should be easy), plus `outputs/decision_boundary.png` (a straight black line
  splitting the two color clusters) and `outputs/loss_curve.png`.
- **Part 2**: scratch and scikit-learn accuracy both land in the mid-to-high 90s on
  breast cancer diagnosis, close but not identical (sklearn's default solver isn't
  plain gradient descent, and it applies L2 regularization by default — see
  exercise 3). Precision/recall/F1 are printed too; project 03 explains what to do
  with them.
- **Part 3 — the important one**: both models start at identical, deliberately bad
  weights. The cross-entropy model's loss should collapse from ~24 to ~0.04 within
  ~300 iterations. **The MSE model's loss should barely move at all** — it gets
  stuck near its starting value. That's not a bug in the MSE branch; it's the
  vanishing-gradient failure mode the math predicted: the `y_hat*(1-y_hat)` factor in
  MSE's gradient is close to 0 exactly when the model is confidently wrong, so the
  update step is tiny exactly when it needs to be large. `outputs/bce_vs_mse_convergence.png`
  plots both, normalized to their own starting loss, on one axis.

## 8. Exercises

1. **Find the tipping point.** In `run_loss_comparison_demo`, the bad init is
   `w=[-8,-8], b=-2`. Try milder bad inits (e.g. `[-2,-2]`) — at what magnitude does
   MSE stop getting stuck and start learning at a comparable rate to cross-entropy?
   This tells you *how* confidently wrong a prediction needs to be before the vanishing
   gradient actually bites — MSE isn't uniformly bad, it's bad specifically near
   saturation.
2. **Move the decision threshold.** In `run_breast_cancer_demo`, change
   `scratch_model.predict(X_test_scaled)` to
   `scratch_model.predict(X_test_scaled, threshold=0.3)`, then try `0.7`. Watch
   precision and recall move in opposite directions. In a cancer-screening context,
   argue on paper which threshold you'd actually want and why (hint: think about which
   kind of mistake — false positive vs. false negative — is more costly here).
3. **Add L2 regularization.** Add `+ alpha * self.weights` to `dw` inside `fit()` (the
   gradient of adding `alpha * sum(w^2)` to the loss — same idea as project 01
   exercise 4). Try `alpha=0.1` and `alpha=10.0` on the breast cancer data and watch
   how close the scratch model's accuracy gets to sklearn's default (`LogisticRegression`
   applies L2 regularization automatically, which is one reason your unregularized
   scratch model and sklearn didn't match exactly in Part 2).
4. **Break the boundary.** In `make_synthetic_2d`, move the two cluster centers closer
   together (e.g. `[-1,-1]` and `[1,1]` instead of `[-2,-2]`/`[2,2]`) so the classes
   overlap. Re-run Part 1 — accuracy should drop and the decision boundary plot should
   show visibly misclassified points on the wrong side of the line. This is what
   "irreducible error" looks like for classification: no linear boundary can perfectly
   separate overlapping classes, the same way no line perfectly fit the noisy data in
   project 01.

## 9. What's next

Both projects so far reported accuracy/MSE/R² almost as an afterthought. Project 03
makes evaluation the main subject: train/validation/test splits and *why* a single
split can lie to you, k-fold cross-validation, the bias-variance tradeoff, and a
careful look at precision/recall/F1/ROC-AUC — including cases (like this project's
exercise 2) where "highest accuracy" is the wrong thing to optimize for.
