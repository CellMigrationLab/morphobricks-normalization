"""MorphoBricks code standards that Ruff does not check, as AST checks (spec §9).

The shared copy of this file is identical in every MorphoBricks repository (a SHA-256 test
keeps it so); each repository configures it in ``pyproject.toml`` under
``[tool.morphobricks.code-rules]``.

Ruff (configured in ``pyproject.toml``) checks line length, annotations, docstrings,
complexity, statements and blind or silent exception handling. This module checks the
rest:

- at most 600 lines per file, 60 executable lines per function (docstring excluded),
  eight named parameters per function (``self``/``cls`` excluded);
- no ``contextlib.suppress``, no identical function bodies, no import cycles;
- a docstring (at least an imperative line) on private functions over 10 lines;
- layers: every module is ``pure``, ``io``, ``ui`` or ``orchestration``; pure modules do
  no I/O and no UI, and UI modules do not compute;
- ownership: :data:`RESERVED_NAMES` may only be defined in their owner repository.

Existing violations are listed in a baseline that may only shrink
(:func:`compare_with_baseline`), and the ``# noqa`` comments of the ratcheted Ruff rules
are counted against it (:func:`noqa_counts`).
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

MAX_FILE_LINES = 600
MAX_FUNCTION_LINES = 60
MAX_PARAMETERS = 8
PRIVATE_DOCSTRING_LINES = 10
"""Private functions longer than this many executable lines need a docstring."""
# Bodies shorter than this many statements are too small to count as duplicated logic.
MIN_DUPLICATE_STATEMENTS = 3

FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


def _functions(tree: ast.AST) -> Iterator[FunctionNode]:
    """Yield every function and method in ``tree``, nested ones included."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _body_without_docstring(function: FunctionNode) -> list[ast.stmt]:
    """Return the function's statements after its docstring."""
    body = function.body
    first = body[0]
    is_docstring = (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    )
    return body[1:] if is_docstring else body


def _executable_lines(function: FunctionNode, lines: list[str]) -> int:
    """Count non-blank, non-comment source lines of the body after the docstring."""
    body = _body_without_docstring(function)
    if not body:
        return 0
    start, end = body[0].lineno, max(stmt.end_lineno or stmt.lineno for stmt in body)
    code = (line.strip() for line in lines[start - 1 : end])
    return sum(1 for line in code if line and not line.startswith("#"))


def _parameter_count(function: FunctionNode, method: bool) -> int:
    """Count named parameters, excluding a method's ``self``/``cls``."""
    arguments = [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]
    if method and arguments and arguments[0].arg in {"self", "cls"}:
        arguments = arguments[1:]
    return len(arguments)


def _methods(tree: ast.AST) -> set[ast.AST]:
    """Return functions defined directly in a class body."""
    return {
        item
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        for item in node.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _uses_suppress(tree: ast.AST) -> bool:
    """Return whether the module refers to ``contextlib.suppress`` in any form."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "suppress":
            return True
        if isinstance(node, ast.ImportFrom) and node.module == "contextlib":
            if any(alias.name == "suppress" for alias in node.names):
                return True
    return False


def check_source(text: str, name: str) -> list[str]:
    """Return every rule violation of one module's source.

    Parameters
    ----------
    text
        Python source.
    name
        Name used in the messages, usually the relative path.

    Returns
    -------
    list of str
        One message per violation; empty when the module follows the rules.
    """
    lines = text.splitlines()
    tree = ast.parse(text, filename=name)
    problems = []
    if len(lines) > MAX_FILE_LINES:
        problems.append(f"{name}: {len(lines)} lines (max {MAX_FILE_LINES})")
    methods = _methods(tree)
    for function in _functions(tree):
        where = f"{name}:{function.lineno} {function.name}"
        length = _executable_lines(function, lines)
        if length > MAX_FUNCTION_LINES:
            problems.append(f"{where}: {length} executable lines (max {MAX_FUNCTION_LINES})")
        count = _parameter_count(function, function in methods)
        if count > MAX_PARAMETERS:
            problems.append(f"{where}: {count} parameters (max {MAX_PARAMETERS})")
    if _uses_suppress(tree):
        problems.append(f"{name}: contextlib.suppress hides exceptions; handle them explicitly")
    return problems


def duplicate_bodies(sources: Iterable[tuple[str, str]]) -> list[str]:
    """Return functions whose bodies (docstring excluded) are identical.

    Parameters
    ----------
    sources
        ``(name, source)`` pairs of the modules to compare.

    Returns
    -------
    list of str
        One message per group of identical bodies.
    """
    seen: dict[str, list[str]] = {}
    for name, text in sources:
        for function in _functions(ast.parse(text, filename=name)):
            body = _body_without_docstring(function)
            if len(body) >= MIN_DUPLICATE_STATEMENTS:
                dump = "\n".join(ast.dump(statement) for statement in body)
                seen.setdefault(dump, []).append(f"{name}:{function.lineno} {function.name}")
    return [f"identical bodies: {', '.join(where)}" for where in seen.values() if len(where) > 1]


def _imported_modules(tree: ast.AST, package: str) -> set[str]:
    """Return the modules of ``package`` that a module imports."""
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    return {module for module in imported if module.split(".")[0] == package}


def _cycle_from(start: str, graph: dict[str, set[str]]) -> list[str]:
    """Return one import cycle through ``start``, or an empty list."""
    stack = [(start, [start])]
    while stack:
        module, path = stack.pop()
        for target in sorted(graph.get(module, ())):
            if target == start:
                return [*path, start]
            if target not in path:
                stack.append((target, [*path, target]))
    return []


def import_cycles(package_dir: Path) -> list[str]:
    """Return import cycles among the modules of one package.

    Parameters
    ----------
    package_dir
        Directory of the package, such as ``src/morphofeature``.

    Returns
    -------
    list of str
        One message per module that lies on a cycle.
    """
    package = package_dir.name
    graph = {}
    for path in sorted(package_dir.rglob("*.py")):
        parts = path.relative_to(package_dir.parent).with_suffix("").parts
        module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        graph[module] = _imported_modules(tree, package) - {module}
    # The package's __init__ re-exports its modules; only cycles among them matter.
    graph.pop(package, None)
    cycles = (_cycle_from(module, graph) for module in sorted(graph))
    return [f"import cycle: {' -> '.join(cycle)}" for cycle in cycles if cycle]


RESERVED_NAMES: Mapping[str, str] = {
    # morphobricks-normalization: histograms, the table, pooling and normalization.
    "compute_plane_histogram": "normalization",
    "compute_block_histogram": "normalization",
    "histogram_with_range_check": "normalization",
    "PooledHistogram": "normalization",
    "pooled_statistics": "normalization",
    "NormalizationParameters": "normalization",
    "min_max": "normalization",
    "percentile_range": "normalization",
    "z_score": "normalization",
    "robust_z_score": "normalization",
    "make_plane_id": "normalization",
    "write_plane_histograms": "normalization",
    "read_plane_histograms": "normalization",
    "image_summaries": "normalization",
    "intensity_checks": "normalization",
    # morphobricks-preprocessing: identities, axes, units, labels, masks, sources.
    "check_identity_text": "preprocessing",
    "check_identity_case": "preprocessing",
    "check_plane_index": "preprocessing",
    "position_id": "preprocessing",
    "named_axes": "preprocessing",
    "header_axes": "preprocessing",
    "space_unit": "preprocessing",
    "to_micrometers": "preprocessing",
    "same_pixel_size": "preprocessing",
    "pixel_size_groups": "preprocessing",
    "check_labels": "preprocessing",
    "find_masks": "preprocessing",
    "read_mask": "preprocessing",
    "read_confirmed_design": "preprocessing",
    "source_fingerprint": "preprocessing",
    "check_sources": "preprocessing",
    "discover_experiment": "preprocessing",
    # CellPatcher: patches and their table.
    "object_sides": "cellpatcher",
    "check_patch_bounds": "cellpatcher",
    "read_patch_table": "cellpatcher",
}
"""Names of shared logic, by the repository that owns it (spec §9, "Ownership"). A function
or class of one of these names defined in any other repository is a copy, and fails."""

LAYERS = ("pure", "io", "ui", "orchestration")

IO_MODULES = frozenset(
    {"os", "shutil", "tempfile", "subprocess", "tifffile", "zarr", "numcodecs", "socket"}
)
"""Modules whose import makes a module do I/O."""

UI_MODULES = frozenset({"ipywidgets", "matplotlib", "IPython", "tkinter"})
"""Modules whose import makes a module a user interface."""

COMPUTE_MODULES = frozenset({"scipy"})
"""Modules a UI module may not import: UI modules show results, they do not compute them."""

IO_CALLS = frozenset(
    {
        "open",
        "print",
        "read_text",
        "write_text",
        "read_bytes",
        "write_bytes",
        "mkdir",
        "unlink",
        "rmdir",
        "iterdir",
        "rglob",
        "glob",
        "stat",
    }
)
"""Calls (by name or attribute) that a pure module may not make."""

RATCHETED_NOQA = ("C901", "PLR0913", "PLR0915", "ANN")
"""Ruff rules whose ``# noqa`` comments are counted against the baseline."""


def missing_private_docstrings(text: str, name: str) -> list[str]:
    """Return private functions over :data:`PRIVATE_DOCSTRING_LINES` lines without a docstring.

    Public names are Ruff's (``D`` rules); a private function needs at least an imperative
    line once it is longer than a few lines.

    Parameters
    ----------
    text
        Python source.
    name
        Name used in the messages, usually the relative path.

    Returns
    -------
    list of str
        One message per function.
    """
    lines = text.splitlines()
    problems = []
    for function in _functions(ast.parse(text, filename=name)):
        private = function.name.startswith("_") and not function.name.startswith("__")
        if not private or ast.get_docstring(function) is not None:
            continue
        length = _executable_lines(function, lines)
        if length > PRIVATE_DOCSTRING_LINES:
            problems.append(
                f"{name}:{function.lineno} {function.name}: private function of {length} "
                "lines without a docstring"
            )
    return problems


def _imports(tree: ast.AST) -> set[str]:
    """Return the top-level names of every module a module imports (absolute imports)."""
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found


def _called_names(tree: ast.AST) -> set[str]:
    """Return the names of every call: ``f(...)`` gives ``f``, ``a.b(...)`` gives ``b``."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
    return names


def layer_problems(text: str, name: str, layer: str | None) -> list[str]:
    """Return the layer violations of one module.

    Parameters
    ----------
    text
        Python source.
    name
        Name used in the messages, usually the relative path.
    layer
        The module's declared layer (:data:`LAYERS`), or None when it has none.

    Returns
    -------
    list of str
        A missing or unknown layer; I/O or UI in a pure module; computing imports in a UI
        module.
    """
    if layer not in LAYERS:
        return [f"{name}: no layer declared (one of {', '.join(LAYERS)})"]
    tree = ast.parse(text, filename=name)
    imports = _imports(tree)
    problems = []
    if layer == "pure":
        for module in sorted(imports & (IO_MODULES | UI_MODULES)):
            problems.append(f"{name}: pure module imports {module}")
        for call in sorted(_called_names(tree) & IO_CALLS):
            problems.append(f"{name}: pure module calls {call}()")
    if layer == "ui":
        for module in sorted(imports & COMPUTE_MODULES):
            problems.append(f"{name}: UI module imports {module}")
    return problems


def reserved_name_problems(text: str, name: str, owner: str) -> list[str]:
    """Return definitions of :data:`RESERVED_NAMES` that this repository does not own.

    Parameters
    ----------
    text
        Python source.
    name
        Name used in the messages, usually the relative path.
    owner
        This repository's owner name (``"preprocessing"``, ``"normalization"``, ...).

    Returns
    -------
    list of str
        One message per copied definition.
    """
    problems = []
    for node in ast.walk(ast.parse(text, filename=name)):
        defined = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        if defined and RESERVED_NAMES.get(node.name, owner) != owner:
            problems.append(
                f"{name}:{node.lineno} {node.name}: owned by {RESERVED_NAMES[node.name]}; "
                "call it there instead of copying it"
            )
    return problems


def noqa_counts(sources: Iterable[tuple[str, str]]) -> dict[str, int]:
    """Count the ``# noqa`` codes of :data:`RATCHETED_NOQA` in the sources.

    Parameters
    ----------
    sources
        ``(name, source)`` pairs.

    Returns
    -------
    dict
        ``{code prefix: count}`` for every ratcheted prefix (0 when none).
    """
    counts = dict.fromkeys(RATCHETED_NOQA, 0)
    for _, text in sources:
        # Line comments and file-level ones (``ruff: noqa: ANN001`` after the hash) count.
        for comment in re.findall(r"#\s*(?:(?:ruff|flake8):\s*)?noqa:\s*([A-Z0-9, ]+)", text):
            for code in (part.strip() for part in comment.split(",")):
                prefix = next((p for p in RATCHETED_NOQA if code.startswith(p)), None)
                if prefix is not None:
                    counts[prefix] += 1
    return counts


def baseline_key(problem: str) -> tuple[str, int]:
    """Split a violation into a stable key and its size.

    Line numbers are dropped and the first remaining number (a length or count) becomes the
    size, so a violation keeps its key while its code moves or shrinks. A line number
    (``path:12``) or a digit inside a name is never a size: a message without a length or
    count, such as identical bodies or an import cycle, has size 0.

    Parameters
    ----------
    problem
        A message of the checks above.

    Returns
    -------
    tuple
        ``(key, size)``; the size is 0 when the message has no number.
    """
    text = re.sub(r"^([^:]+):\d+ ", r"\1 ", problem)
    found = re.search(r"(?<![\w.:])\d+(?![\w.])", text.split(": ", 1)[-1])
    size = int(found.group()) if found else 0
    key = re.sub(r"\d+", "#", text)
    return key, size


def compare_with_baseline(
    problems: Iterable[str], noqa: Mapping[str, int], baseline: Mapping
) -> list[str]:
    """Return what breaks the baseline: new violations, growth, and entries that can shrink.

    The baseline may only shrink: a violation not in it, or larger than recorded, fails;
    so does an entry that is no longer needed (or larger than needed), until it is lowered
    with ``tools/check.py --update-baseline``. ``# noqa`` counts work the same way.

    Parameters
    ----------
    problems
        The current violations.
    noqa
        The current :func:`noqa_counts`.
    baseline
        ``{"problems": {key: size}, "noqa": {prefix: count}}``.

    Returns
    -------
    list of str
        One message per difference; empty when the code matches the baseline.
    """
    allowed = dict(baseline.get("problems", {}))
    current: dict[str, int] = {}
    for problem in problems:
        key, size = baseline_key(problem)
        current[key] = max(current.get(key, 0), size)
    found = []
    for key, size in sorted(current.items()):
        if key not in allowed:
            found.append(f"new violation: {key}")
        elif size > allowed[key]:
            found.append(f"grew from {allowed[key]} to {size}: {key}")
    for key, size in sorted(allowed.items()):
        if key not in current or current[key] < size:
            found.append(f"baseline can shrink (run tools/check.py --update-baseline): {key}")
    for prefix, count in sorted(noqa.items()):
        limit = baseline.get("noqa", {}).get(prefix, 0)
        if count > limit:
            found.append(f"# noqa: {prefix} grew from {limit} to {count}")
        elif count < limit:
            found.append(
                f"# noqa: {prefix} can shrink from {limit} to {count} "
                "(run tools/check.py --update-baseline)"
            )
    return found


def shrunk_baseline(problems: Iterable[str], noqa: Mapping[str, int], baseline: Mapping) -> dict:
    """Return the baseline lowered to the current code, never raised.

    Parameters
    ----------
    problems
        The current violations.
    noqa
        The current :func:`noqa_counts`.
    baseline
        The recorded baseline.

    Returns
    -------
    dict
        The new baseline: entries no longer violated removed, sizes and counts lowered; new
        or grown violations are not added.
    """
    allowed = dict(baseline.get("problems", {}))
    current: dict[str, int] = {}
    for problem in problems:
        key, size = baseline_key(problem)
        current[key] = max(current.get(key, 0), size)
    kept = {k: min(v, current[k]) for k, v in allowed.items() if k in current}
    limits = baseline.get("noqa", {})
    return {
        "problems": dict(sorted(kept.items())),
        "noqa": {p: min(limits.get(p, 0), noqa.get(p, 0)) for p in RATCHETED_NOQA},
    }


def read_baseline(path: Path) -> dict:
    """Read a baseline file; an absent file is an empty baseline.

    Parameters
    ----------
    path
        The JSON file.

    Returns
    -------
    dict
        ``{"problems": {...}, "noqa": {...}}``.
    """
    if not path.exists():
        return {"problems": {}, "noqa": {}}
    return json.loads(path.read_text(encoding="utf-8"))
