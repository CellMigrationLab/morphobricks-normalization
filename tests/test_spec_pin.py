"""Pin integrity of the vendored MorphoBricks-Spec files (offline, no git, no network).

TEMPLATE: copy this file to ``tests/test_spec_pin.py`` of a brick. It is maintained in
MorphoBricks-Spec (``tools/brick_templates/``); change it there. Docs: ``docs/vendoring.md``.

The vendored folder (``tests/spec_fixtures`` by default, made by ``tools/vendor.py``) must hash
to its ``PIN.toml``: ``fixtures/`` to the git tree-object ID ``git rev-parse <commit>:fixtures``
and every other pinned file to its git blob ID. One changed, added or missing byte fails a
test, so a local edit of a fixture cannot make a reader pass. The hashing is git's own (the
scheme of the preprocessing brick's ``tests/spec_pin.py``), computed here with ``hashlib``.

Edit only ``VENDORED`` below if the copy lives elsewhere. Needs Python 3.11 or newer
(``tomllib``) and pytest. These tests never skip. ``MORPHOBRICKS_SPEC_ROOT`` points them at
another folder laid out like the vendored one.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tomllib
from pathlib import Path

import pytest

VENDORED = Path(__file__).parent / "spec_fixtures"
ROOT_ENV = "MORPHOBRICKS_SPEC_ROOT"
#: Files of the folder that belong to the copy but are not Spec content.
OWN_FILES = frozenset({"PIN.toml", "README.md", ".gitattributes"})


class SpecPinError(Exception):
    """Raised when the vendored Spec files are missing, unreadable or malformed."""


def spec_root() -> Path:
    """Return the folder holding ``fixtures/``, ``docs/`` and ``PIN.toml``."""
    override = os.environ.get(ROOT_ENV)
    return Path(override) if override else VENDORED


def read_pin(root: Path | None = None) -> dict:
    """Return the parsed ``PIN.toml``, or raise ``SpecPinError`` when it is absent."""
    path = (root or spec_root()) / "PIN.toml"
    if not path.is_file():
        raise SpecPinError(f"{path} is missing: the Spec files are not vendored here")
    return tomllib.loads(path.read_text(encoding="utf-8"))


def git_blob_id(data: bytes) -> str:
    """Return the git blob-object ID of ``data`` (``git hash-object``)."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def git_tree_id(folder: Path) -> str:
    """Return the git tree-object ID of a folder, as ``git write-tree`` would give it.

    Every file counts as mode 100644 (the vendored copy has no executable files, and a
    file system without mode bits could not tell). A symbolic link, a missing folder or
    anything that is neither a file nor a folder raises ``SpecPinError``.
    """
    if not folder.is_dir() or folder.is_symlink():
        raise SpecPinError(f"{folder} is not a folder")
    entries = []
    for entry in os.scandir(folder):
        name = entry.name.encode("utf-8")
        if entry.is_symlink():
            raise SpecPinError(f"{entry.path} is a symbolic link")
        if entry.is_dir():
            entries.append((name + b"/", b"40000", name, bytes.fromhex(git_tree_id(Path(entry)))))
        elif entry.is_file():
            blob = git_blob_id(Path(entry).read_bytes())
            entries.append((name, b"100644", name, bytes.fromhex(blob)))
        else:
            raise SpecPinError(f"{entry.path} is neither a file nor a folder")
    body = b"".join(mode + b" " + name + b"\0" + sha for _, mode, name, sha in sorted(entries))
    return hashlib.sha1(b"tree %d\0" % len(body) + body).hexdigest()


def pin_problems(root: Path | None = None) -> list[str]:
    """Return what differs between the vendored files and the pin (empty when they match)."""
    root = root or spec_root()
    pin = read_pin(root)
    problems = []
    if not re.fullmatch(r"[0-9a-f]{40}", str(pin.get("commit", ""))):
        problems.append("the pin is not a full 40-digit commit SHA")
    tree = git_tree_id(root / "fixtures")
    if tree != pin.get("fixtures_tree"):
        problems.append(f"fixtures/ hashes to {tree}, the pin says {pin.get('fixtures_tree')}")
    for relative, blob in pin.get("files", {}).items():
        path = root / relative
        if not path.is_file():
            problems.append(f"{relative} is missing")
        elif git_blob_id(path.read_bytes()) != blob:
            problems.append(f"{relative} does not hash to its pin {blob}")
    allowed = OWN_FILES | set(pin.get("files", {}))
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if "__pycache__" in path.parts or not path.is_file() or relative.startswith("fixtures/"):
            continue
        if relative not in allowed:
            problems.append(f"{relative} is not a pinned file")
    return problems


def test_vendored_files_hash_to_the_pin():
    assert pin_problems() == []


def test_the_pin_is_a_full_commit_sha_never_a_branch():
    pin = read_pin()
    assert re.fullmatch(r"[0-9a-f]{40}", pin["commit"])
    assert re.fullmatch(r"[0-9a-f]{40}", pin["fixtures_tree"])
    assert pin["files"], "the pin names no file"
    assert all(re.fullmatch(r"[0-9a-f]{40}", blob) for blob in pin["files"].values())


def test_the_pinned_tree_is_the_folder_that_is_vendored():
    # The expected value is the pin's own: this fails when fixtures/ is not what it names.
    assert git_tree_id(spec_root() / "fixtures") == read_pin()["fixtures_tree"]


def test_git_blob_and_tree_ids_equal_git_s():
    # Known answers computed with git itself (git hash-object, git write-tree).
    assert git_blob_id(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_git_tree_id_orders_a_folder_as_its_name_and_a_slash(tmp_path):
    # `a.b` sorts before the folder `a` (compared as `a/`), `a-b` before both: the order that
    # makes a hand-rolled tree hash differ from git's when it sorts by plain name.
    (tmp_path / "b").mkdir()
    (tmp_path / "a").mkdir()
    (tmp_path / "a.txt").write_bytes(b"hello\n")
    (tmp_path / "b" / "c").write_bytes(b"x")
    (tmp_path / "a.b").write_bytes(b"y")
    (tmp_path / "a-b").write_bytes(b"z")
    (tmp_path / "a" / "f").write_bytes(b"w")
    assert git_tree_id(tmp_path) == "fec6a54403e1c2a94724c1aab9f127c686f6dbb0"


def _copy(tmp_path: Path) -> Path:
    root = tmp_path / "spec"
    shutil.copytree(spec_root(), root)
    return root


def _first(root: Path, *parts: str) -> Path:
    files = sorted(p for p in (root.joinpath(*parts)).rglob("*") if p.is_file())
    assert files, f"nothing under {'/'.join(parts)}"
    return files[0]


def _flip_byte(path: Path) -> None:
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x01
    path.write_bytes(bytes(data))


def _flip_fixture(root: Path) -> None:
    _flip_byte(_first(root, "fixtures", "valid"))


def _add_fixture(root: Path) -> None:
    (root / "fixtures" / "valid" / "v99_added.toml").write_bytes(b"# added\n")


def _remove_fixture(root: Path) -> None:
    _first(root, "fixtures", "invalid").unlink()


def _rename_fixture(root: Path) -> None:
    path = _first(root, "fixtures", "valid")
    path.rename(path.with_name("renamed_" + path.name))


def _crlf_fixture(root: Path) -> None:
    path = _first(root, "fixtures", "valid")
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))


def _flip_pinned_file(root: Path) -> None:
    _flip_byte(root / sorted(read_pin(root)["files"])[0])


def _remove_pinned_file(root: Path) -> None:
    (root / sorted(read_pin(root)["files"])[0]).unlink()


def _add_stray_file(root: Path) -> None:
    (root / "docs").mkdir(exist_ok=True)
    (root / "docs" / "stray.md").write_bytes(b"not pinned\n")


MUTATIONS = [
    _flip_fixture,
    _add_fixture,
    _remove_fixture,
    _rename_fixture,
    _crlf_fixture,
    _flip_pinned_file,
    _remove_pinned_file,
    _add_stray_file,
]


@pytest.mark.parametrize("mutate", MUTATIONS, ids=lambda f: f.__name__.lstrip("_"))
def test_one_changed_added_or_missing_byte_fails_the_pin(tmp_path, mutate):
    root = _copy(tmp_path)
    assert pin_problems(root) == []
    mutate(root)
    assert pin_problems(root) != []


def test_a_symbolic_link_in_the_fixtures_is_refused(tmp_path):
    root = _copy(tmp_path)
    link = root / "fixtures" / "valid" / "v98_link.toml"
    try:
        link.symlink_to(_first(root, "fixtures", "valid"))
    except OSError as error:  # a file system without links: report it, never skip
        pytest.fail(f"cannot create a link to test the refusal: {error}")
    with pytest.raises(SpecPinError, match="symbolic link"):
        pin_problems(root)


def test_a_pin_that_is_not_a_full_sha_is_reported(tmp_path):
    root = _copy(tmp_path)
    pin = root / "PIN.toml"
    pin.write_text(pin.read_text(encoding="utf-8").replace(read_pin()["commit"], "main"))
    assert any("40-digit" in problem for problem in pin_problems(root))


def test_a_missing_pin_file_is_an_error_not_a_pass(tmp_path):
    with pytest.raises(SpecPinError, match="PIN.toml is missing"):
        pin_problems(tmp_path)
