# One-time environment setup

Each project has its own `requirements.txt` (so later projects can add `torch`,
`langchain`, `anthropic`, etc. without bloating the earlier ones), but they all share
the same core stack for phase 1 projects: `numpy`, `pandas`, `matplotlib`,
`scikit-learn`.

You need Python 3.10+ (check with `python3 --version`).

## Per-project virtual environment (recommended)

Run this inside each project folder the first time you open it:

```bash
cd 01-linear-regression-from-scratch   # or whichever project
python3 -m venv .venv
source .venv/bin/activate              # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Each project's README repeats this so you never have to come back here — but if you'd
rather have one shared venv for the whole repo instead of one per folder, that also
works fine: create it at the repo root and `pip install -r <project>/requirements.txt`
before running each project.

## Deactivating

```bash
deactivate
```

## macOS "certificate verify failed" when downloading datasets

If a project downloads a dataset (e.g. `sklearn.datasets.fetch_california_housing`)
and you see `SSL: CERTIFICATE_VERIFY_FAILED`, that's a known python.org-installer-on-
macOS issue — Python ships its own CA bundle but doesn't wire it up automatically.
Two fixes, either works:

```bash
# Option A: run the installer script python.org put in your Applications folder
open "/Applications/Python 3.11/Install Certificates.command"   # match your version

# Option B: point this project's venv at certifi's CA bundle for this run only
pip install certifi
export SSL_CERT_FILE=$(python3 -c "import certifi; print(certifi.where())")
python your_script.py
```

## GenAI projects (11+)

These additionally need an LLM API key. See the root `README.md` section "A note on
the GenAI projects" and each project's own README for exact setup — you don't need to
deal with this until you get there.
