# 01 — Linear Regression from Scratch

## 1. What you'll build

A linear regression model implemented with nothing but numpy — no `sklearn.fit()`
doing the work for you. You'll train it two ways (iterative gradient descent, and the
closed-form "normal equation"), run both on synthetic data and on a real dataset
(California housing prices), and check your from-scratch numbers against
scikit-learn's implementation to prove your math is right.

By the end you'll be able to explain, from the formula up, *why* gradient descent
works, not just that it does.

## 2. The core idea

You have some inputs (square footage, income, whatever) and a number you want to
predict (house price). Linear regression's bet: the output is (approximately) a
weighted sum of the inputs plus a constant offset. "Training" means finding the
weights that make that sum match reality as closely as possible, on average, across
all your examples.

"As closely as possible" needs a precise definition — that's the loss function. And
"find the weights" needs an algorithm — that's gradient descent (or, for this simple a
model, algebra).

## 3. The math

**The model** — a prediction is a dot product plus a bias:

```
y_hat = X @ w + b
```

`X` is an `(n_samples, n_features)` matrix, `w` is a weight per feature, `b` is a
single scalar offset (the "intercept" — the prediction when every feature is 0).

**The loss** — Mean Squared Error:

```
J(w, b) = (1/n) * Σ (y_hat_i - y_i)^2
```

Why squared, not absolute difference? Two reasons: (a) it's differentiable everywhere
(absolute value has a kink at 0, which complicates gradient-based optimization), and
(b) it penalizes large errors disproportionately more than small ones, which is
usually what you want — being off by 10 should hurt more than 10x being off by 1, not
the same. (Project 02 shows a case, classification, where squared error is actually
the *wrong* choice, and why.)

**The gradient** — to minimize `J`, we need to know which direction to move `w` and
`b` to make it smaller. That's the gradient, `∂J/∂w`. Deriving it via the chain rule,
for a single weight `w_j`:

```
∂J/∂w_j = (1/n) * Σ 2(y_hat_i - y_i) * ∂y_hat_i/∂w_j
        = (1/n) * Σ 2(y_hat_i - y_i) * x_ij          [since y_hat_i = Σ w_k x_ik + b, ∂y_hat_i/∂w_j = x_ij]
        = (2/n) * Σ (y_hat_i - y_i) * x_ij
```

Stacked across all features, that's a single matrix expression:

```
∂J/∂w = (2/n) * X^T @ (y_hat - y)
∂J/∂b = (2/n) * Σ (y_hat - y)
```

**The update rule** — gradient descent: repeatedly step *opposite* the gradient
(downhill on the loss surface), scaled by a learning rate `α`:

```
w := w - α * ∂J/∂w
b := b - α * ∂J/∂b
```

Do this enough times and `w, b` converge to (approximately) whatever values minimize
`J`. Too big an `α` and you overshoot and diverge; too small and it takes forever —
you'll see this in the exercises.

**The closed-form alternative** — because MSE is a convex quadratic in `w`, you can
skip the iteration entirely and solve for where the gradient is exactly zero:

```
∂J/∂w = 0
(2/n) * X^T (Xw - y) = 0
X^T X w = X^T y
w = (X^T X)^-1 X^T y          <- the "normal equation"
```

(Bias is folded into `w` by prepending a column of 1s to `X`.) This gives the exact
optimum in one shot — no learning rate, no iterations. The catch: it requires
inverting a `(features × features)` matrix, which costs `O(d^3)` and needs `X^T X` to
be invertible. For a handful of features that's instant; for thousands of features (or
models with no closed form at all, like neural networks — see project 06) it's
infeasible, and gradient descent is the only option. Learning both here is the point:
you'll use gradient descent for the rest of this curriculum, but it's worth seeing
once that it's approximating something that, in this one simple case, has an exact
answer you can check against.

## 4. From formula to code

Open [`linear_regression.py`](linear_regression.py) — the `LinearRegressionGD` class
docstring numbers each formula `(1)`–`(5)`, and the matching line in `fit()` is tagged
with the same number:

| Formula | Code |
|---|---|
| `y_hat = X @ w + b` | `y_pred = X @ self.weights + self.bias` |
| `J = mean((y_hat - y)^2)` | `np.mean(error ** 2)` |
| `∂J/∂w = (2/n) X^T(y_hat - y)` | `dw = (2 / n_samples) * (X.T @ error)` |
| `∂J/∂b = (2/n) Σ(y_hat - y)` | `db = (2 / n_samples) * np.sum(error)` |
| `w := w - α·∂J/∂w` | `self.weights -= self.learning_rate * dw` |
| `w = (X^T X)^-1 X^T y` | `normal_equation()` function |

## 5. The data

Two datasets, deliberately different in shape:

1. **Synthetic 1D data** (`run_synthetic_demo`): points generated from a known line
   `y = 3.5x - 2` plus random noise. Because we *know* the true function, we can check
   that gradient descent actually recovers something close to `w=3.5, b=-2` — this is
   the fastest way to sanity-check an implementation before trusting it on real data
   where you don't know the "true" answer.
2. **California housing** (`run_california_housing_demo`): a real dataset built into
   scikit-learn (`fetch_california_housing`), 8 features (median income, house age,
   average rooms, population, location, ...) predicting median house value per
   district. This is where you see linear regression on data with multiple features
   of very different scales — which is why feature standardization shows up here and
   didn't in the 1D case.

## 6. Build it

Everything's in [`linear_regression.py`](linear_regression.py):
- `LinearRegressionGD` — the from-scratch model (fit via gradient descent, predict).
- `normal_equation()` — the closed-form solver.
- `run_synthetic_demo()` — 1D fit + loss curve + fitted-line plot.
- `run_california_housing_demo()` — 3-way comparison table on real data.

## 7. Train & evaluate

Set up the environment and run it:

```bash
cd 01-linear-regression-from-scratch
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python linear_regression.py
```

(If Part 2 fails with `SSL: CERTIFICATE_VERIFY_FAILED` while downloading the housing
dataset — a common macOS/python.org quirk — see the fix in `../_shared/setup.md`.)

Expect:
- Part 1 prints the learned `w, b` next to the true `w=3.5, b=-2` (they should be
  close but not identical — noise means the training sample never exactly matches the
  underlying function), plus saves `outputs/loss_curve.png` (should monotonically
  decrease and flatten — that flattening *is* convergence) and
  `outputs/fitted_line.png`.
- Part 2 prints a table comparing MSE and R² across the three methods. **They should
  all land within a hair of each other** — that agreement is your correctness check.
  If gradient descent's numbers are noticeably worse, something's off (see exercises).

**Why MSE and R² as metrics**: MSE is what we're optimizing, so it directly measures
training success, but it's in squared, hard-to-interpret units (dollars², here). R²
("coefficient of determination") rescales that into "fraction of variance in `y`
explained by the model," from roughly 0 (no better than predicting the mean) to 1
(perfect) — much easier to reason about, and comparable across different datasets.
Project 03 goes deep on evaluation metrics; this is just enough to know if your model
is doing anything useful.

## 8. Exercises

Do these by editing `linear_regression.py` and re-running — the goal is to *feel* the
effect, not just read about it:

1. **Break convergence.** Set `learning_rate=5.0` in `run_synthetic_demo`'s model.
   Watch the loss curve in `loss_curve.png` — instead of decreasing, it should explode
   or oscillate. Now try `learning_rate=0.0001`. What happens to how many iterations
   you'd need? This is the overshoot-vs-crawl tradeoff learning rate always involves.
2. **Skip standardization.** In `run_california_housing_demo`, feed the *unscaled*
   `X_train`/`X_test` into `LinearRegressionGD` instead of the scaled versions (keep
   the normal equation and sklearn on scaled data for comparison). Watch the scratch
   model's MSE — it should get dramatically worse or even diverge to `nan`. Why:
   features on wildly different scales (population in the thousands vs. rooms per
   household in single digits) mean a *single* learning rate is simultaneously too big
   for some weights and too small for others.
3. **Prove the normal equation is exact.** Turn `n_iterations` up to 20,000 in
   `run_california_housing_demo`'s scratch model. Its MSE/R² should converge to
   *exactly* match the normal equation's — because gradient descent, run long enough
   on a convex loss, finds the same unique minimum the algebra finds directly.
4. **Add L2 regularization (ridge regression).** Add `+ alpha * w` to the `dw`
   gradient (this is the gradient of adding `alpha * sum(w^2)` to the loss — work out
   why on paper first). Try `alpha=1.0` and `alpha=100.0` on the housing data and watch
   the learned weights shrink toward zero. This previews why regularization exists:
   it trades a little training accuracy for weights that generalize better and don't
   blow up on collinear features.

## 9. What's next

Project 02 keeps the same gradient-descent machinery but changes the *task*:
predicting a category instead of a number. You'll see exactly why MSE — which worked
fine here — becomes the wrong loss function for classification, and derive
cross-entropy loss and the sigmoid function from first principles the same way we
derived MSE's gradient above.
