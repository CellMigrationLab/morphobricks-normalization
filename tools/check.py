"""Run every local check a pull request must pass: the code rules, Ruff and pytest.

Usage:

- ``python tools/check.py``: everything; exits non-zero when any check fails, after
  running all of them so one run lists every problem;
- ``python tools/check.py --rules``: only the code rules and Ruff, against their baseline
  (CI);
- ``python tools/check.py --update-baseline``: lower the baseline to the current code. It
  never adds or raises an entry: new violations must be fixed.
- ``python tools/check.py --rename OLD=NEW``: re-key the baseline entries of code moved from
  ``OLD`` to ``NEW`` (a file, or a function as ``path name``), for a pull request that only
  moves code; then ``--update-baseline`` drops what the move removed. Nothing is added or
  raised: an entry whose new key already exists keeps the smaller size.

The code rules are the shared ``tools/code_rules.py``, configured in ``pyproject.toml``
under ``[tool.morphobricks.code-rules]`` (owner, package, layers, baseline file).

This runner is fiberfeatures' ``tools/check.py``, unchanged but for this paragraph and the
``ruff check`` and ``ruff format --check`` steps of the full run. normalization adopted the
standards (design v0.4 §9) with existing debt, so Ruff's findings
go through the same shrink-only baseline as the code rules: each is counted per file,
function and rule (:func:`ruff_problems`), and a function may not gain a finding of a rule.
Ruff runs with ``--ignore-noqa``: a ``# noqa`` comment suppresses nothing here, so no rule can be
silenced past the baseline (the shared ``# noqa`` ratchet counts only C901, PLR0913, PLR0915 and
ANN, and Ruff would otherwise honour a new ``# noqa: F401`` without a word).
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path

from code_rules import (
    check_source,
    compare_with_baseline,
    duplicate_bodies,
    import_cycles,
    layer_problems,
    missing_private_docstrings,
    noqa_counts,
    read_baseline,
    reserved_name_problems,
    shrunk_baseline,
)

ROOT = Path(__file__).resolve().parents[1]
LINTED = ("src", "tests", "tools")
TESTS = "tests/"
"""Tests are exempt from the annotation and docstring rules (owner decision, 1 November
2026): Ruff ignores ``ANN`` and ``D`` there (``pyproject.toml``), and the private-docstring
rule leaves them out here. Every other rule applies."""
MODULE_LEVEL = "<module>"
"""The name a Ruff finding outside every function is counted under."""


def config() -> dict:
    """Return this repository's code-rules settings from ``pyproject.toml``."""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["tool"]["morphobricks"]["code-rules"]


def _python_files() -> list[Path]:
    """Return the repository's checked Python files (guard fixtures are deliberately bad)."""
    files = [path for folder in LINTED for path in sorted((ROOT / folder).rglob("*.py"))]
    return [path for path in files if "guard_fixtures" not in path.parts]


def _module_name(path: Path, package_dir: Path) -> str:
    """Return the dotted module name of a file of the package."""
    parts = path.relative_to(package_dir.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _function_spans(text: str) -> list[tuple[int, int, str]]:
    """Return ``(first line, last line, name)`` of every function, innermost last when nested."""
    spans = [
        (node.lineno, node.end_lineno or node.lineno, node.name)
        for node in ast.walk(ast.parse(text))
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    return sorted(spans, key=lambda span: (span[0], -span[1]))


def _enclosing_function(spans: list[tuple[int, int, str]], line: int) -> str:
    """Return the innermost function around *line*, or :data:`MODULE_LEVEL`."""
    names = [name for first, last, name in spans if first <= line <= last]
    return names[-1] if names else MODULE_LEVEL


def ruff_findings() -> list[dict]:
    """Return Ruff's findings on the linted folders (its JSON output), ``# noqa`` ignored."""
    command = [sys.executable, "-m", "ruff", "check", "--no-cache", "--exit-zero", "--ignore-noqa"]
    command += ["--output-format=json", *LINTED]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def ruff_problems(findings: list[dict]) -> list[str]:
    """Return Ruff's findings as baseline problems, one per file, function and rule.

    The message gives the count and the rule's name (never its code, whose digits the
    baseline would read as a size): ``src/x.py f: 3 Ruff findings missing-type-function-argument``.
    """
    counts: Counter[tuple[str, str, str]] = Counter()
    spans: dict[str, list[tuple[int, int, str]]] = {}
    for finding in findings:
        path = Path(finding["filename"])
        name = path.relative_to(ROOT).as_posix()
        if name not in spans:
            spans[name] = _function_spans(path.read_text(encoding="utf-8"))
        function = _enclosing_function(spans[name], finding["location"]["row"])
        counts[name, function, finding["name"]] += 1
    return [
        f"{name} {function}: {count} Ruff findings {rule}"
        for (name, function, rule), count in sorted(counts.items())
    ]


def code_rule_problems() -> tuple[list[str], dict[str, int]]:
    """Return every code-rule violation in the repository and the ``# noqa`` counts."""
    settings = config()
    package_dir = ROOT / settings["package"]
    layers = settings.get("layers", {})
    sources = [
        (str(path.relative_to(ROOT).as_posix()), path.read_text(encoding="utf-8"))
        for path in _python_files()
    ]
    problems = []
    for name, text in sources:
        problems += check_source(text, name)
        if not name.startswith(TESTS):
            problems += missing_private_docstrings(text, name)
        problems += reserved_name_problems(text, name, settings["owner"])
        path = ROOT / name
        if package_dir in path.parents:
            problems += layer_problems(text, name, layers.get(_module_name(path, package_dir)))
    problems += duplicate_bodies(sources)
    problems += import_cycles(package_dir)
    return problems, noqa_counts(sources)


def all_problems() -> tuple[list[str], dict[str, int]]:
    """Return the code-rule and Ruff problems, and the ``# noqa`` counts."""
    problems, noqa = code_rule_problems()
    return problems + ruff_problems(ruff_findings()), noqa


def baseline_problems() -> list[str]:
    """Return what breaks the baseline (empty when the code matches it)."""
    problems, noqa = all_problems()
    baseline = read_baseline(ROOT / config()["baseline"])
    return compare_with_baseline(problems, noqa, baseline)


def update_baseline() -> Path:
    """Lower the baseline file to the current code, never raising an entry."""
    problems, noqa = all_problems()
    path = ROOT / config()["baseline"]
    lowered = shrunk_baseline(problems, noqa, read_baseline(path))
    path.write_text(json.dumps(lowered, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def rename_baseline(renames: list[str]) -> Path:
    """Re-key baseline entries for moved code, as ``OLD=NEW`` key prefixes."""
    path = ROOT / config()["baseline"]
    baseline = read_baseline(path)
    problems = {}
    for key, size in baseline.get("problems", {}).items():
        for rename in renames:
            old, new = rename.split("=", 1)
            if key.startswith(old):
                key = new + key[len(old) :]
        problems[key] = min(size, problems.get(key, size))
    baseline["problems"] = dict(sorted(problems.items()))
    path.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv: list[str]) -> int:
    """Run the checks and return the process exit code."""
    if "--rename" in argv:
        renames = [argv[i + 1] for i, arg in enumerate(argv) if arg == "--rename"]
        print(f"Baseline re-keyed: {rename_baseline(renames)}")
        return 0
    if "--update-baseline" in argv:
        print(f"Baseline lowered: {update_baseline()}")
        return 0
    failed = False
    if "--rules" not in argv:
        commands = (
            [sys.executable, "-m", "ruff", "format", "--check", "."],
            [sys.executable, "-m", "pytest", "-q"],
        )
        for command in commands:
            print("$", " ".join(command[2:]), flush=True)
            failed |= subprocess.run(command, cwd=ROOT, check=False).returncode != 0
    print("$ code rules and ruff check", flush=True)
    problems = baseline_problems()
    for problem in problems:
        print(problem)
    return 1 if failed or problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
