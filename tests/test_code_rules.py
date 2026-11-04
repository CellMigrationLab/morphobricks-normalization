"""The shared MorphoBricks code standards (design v0.4 §9): the file, its baseline, the layers.

``tools/code_rules.py`` is a byte-identical copy of the file every MorphoBricks repository
holds; ``tools/check.py`` runs it with Ruff (``--ignore-noqa``) against a baseline that may only
shrink. normalization adopted the rules with its findings at v0.3.3 in that baseline; new and
moved code complies.
"""

from __future__ import annotations

import hashlib
import json

import pytest

import check
from code_rules import RATCHETED_NOQA, compare_with_baseline, shrunk_baseline

SHARED_SHA256 = "cc04a420a4d25d5ef200bfa370f052d9911786fde0c57385dcaf155716d6e0a2"
"""SHA-256 of the shared ``tools/code_rules.py``, identical in every MorphoBricks repository."""


@pytest.fixture(scope="module")
def current():
    return check.all_problems()


def _baseline():
    return json.loads((check.ROOT / check.config()["baseline"]).read_text(encoding="utf-8"))


def test_the_shared_rules_file_is_the_shared_copy():
    # With LF line endings, as committed (a Windows checkout may convert them).
    data = (check.ROOT / "tools" / "code_rules.py").read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(data).hexdigest() == SHARED_SHA256


def test_the_repository_matches_its_baseline(current):
    """No new violation, none grown, and the baseline as small as the code allows."""
    problems, noqa = current
    baseline = _baseline()
    assert compare_with_baseline(problems, noqa, baseline) == []
    assert shrunk_baseline(problems, noqa, baseline) == baseline


def test_no_ratcheted_noqa():
    assert _baseline()["noqa"] == dict.fromkeys(RATCHETED_NOQA, 0)


def test_a_new_violation_breaks_the_baseline(current):
    problems, noqa = current
    extra = "src/morphobricks_normalization/rgb.py f: 1 Ruff findings any-type"
    assert compare_with_baseline([*problems, extra], noqa, _baseline())


def test_every_package_module_has_a_layer():
    package = check.ROOT / check.config()["package"]
    modules = {check._module_name(path, package) for path in package.rglob("*.py")}
    assert modules == set(check.config()["layers"])
