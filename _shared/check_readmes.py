"""
Repo hygiene checks for the project READMEs. Run from the repo root:

    python3 _shared/check_readmes.py

Exists because two things have silently broken before and neither was visible
from the terminal — only in GitHub's rendered view:

  1. A roadmap row lost its link when a str.replace() quietly failed to match.
  2. A '#' inside a math expression broke KaTeX rendering. The source had it
     escaped as '\\#', but GitHub's markdown parser strips the backslash before
     the math renderer sees it, so escaping is NOT a reliable fix — avoid the
     character entirely.

Checks performed:
  - every built project folder is linked from the roadmap table, and every
    roadmap link resolves to a real directory
  - every embedded image path resolves
  - no math expression contains a character that breaks GitHub's KaTeX
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# '&' and '\\' are legitimate inside \begin{cases}/\begin{aligned}; '#' and a
# bare '%' are not recoverable, since GitHub strips the escaping backslash.
FORBIDDEN_IN_MATH = {"#": "macro parameter '#'", "%": "comment character '%'"}


def check_roadmap() -> list[str]:
    problems = []
    rows = [l for l in (ROOT / "README.md").read_text().splitlines()
            if re.match(r"^\| \d+ \|", l)]
    built = {d.name for d in ROOT.iterdir() if re.match(r"^\d\d-", d.name)}
    linked = set()
    for row in rows:
        num, cell = row.split("|")[1].strip(), row.split("|")[2].strip()
        match = re.search(r"\]\(([^)]+)\)", cell)
        if match:
            target = match.group(1).rstrip("/")
            linked.add(target)
            if not (ROOT / target).is_dir():
                problems.append(f"roadmap row {num}: links to {target}, which does not exist")
        elif any(b.startswith(num.zfill(2) + "-") for b in built):
            problems.append(f"roadmap row {num}: project folder exists but the row is not linked")
    for folder in sorted(built - linked):
        problems.append(f"{folder}: exists but is not linked from the roadmap")
    return problems


def check_images() -> list[str]:
    problems = []
    for readme in sorted(ROOT.glob("*/README.md")):
        for match in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", readme.read_text()):
            if not (readme.parent / match.group(1)).exists():
                problems.append(f"{readme.parent.name}: missing image {match.group(1)}")
    return problems


def check_math() -> list[str]:
    problems = []
    for readme in sorted(ROOT.glob("*/README.md")):
        body = re.sub(r"```.*?```", "", readme.read_text(), flags=re.S)  # ignore code blocks
        for match in re.finditer(r"\$\$(.+?)\$\$|(?<!\$)\$([^$\n]+?)\$(?!\$)", body, flags=re.S):
            expr = match.group(1) or match.group(2)
            for char, name in FORBIDDEN_IN_MATH.items():
                if char in expr:
                    line = body[: match.start()].count("\n") + 1
                    problems.append(
                        f"{readme.parent.name} line ~{line}: {name} in math breaks "
                        f"GitHub rendering -> {expr.strip()[:60]}")
    return problems


if __name__ == "__main__":
    all_problems = check_roadmap() + check_images() + check_math()
    if all_problems:
        print("PROBLEMS FOUND:\n")
        print("\n".join(f"  - {p}" for p in all_problems))
        sys.exit(1)
    n = len(list(ROOT.glob("[0-9][0-9]-*")))
    print(f"All checks passed: {n} projects, roadmap links, images and math all valid.")
