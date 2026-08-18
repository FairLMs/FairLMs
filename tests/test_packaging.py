"""Packaging guards.

`fairllms.metrics` eagerly imports every metric family, so any third-party
module imported at module scope under `fairllms/definition/` becomes a hard
requirement of `import fairllms`. A dependency that is only listed in an extra
therefore breaks a plain `pip install fairllms` — which is exactly what happened
with `wordfreq`, `nltk` and `scikit-learn` before 0.2.0.

This test catches that class of bug from the source tree, without needing a
clean-environment install.
"""

from __future__ import annotations

import ast
import pathlib
import shlex
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE = REPO_ROOT / "fairllms"

# import name -> distribution name, where they differ
DIST_NAME = {
    "sklearn": "scikit-learn",
    "gender_guesser": "gender-guesser",
    "yaml": "pyyaml",
    "PIL": "pillow",
}


def _declared_core_dependencies() -> set[str]:
    """Distribution names in [project.dependencies], normalized."""
    tomllib = pytest.importorskip(
        "tomllib", reason="needs Python 3.11+ to read pyproject.toml"
    )
    with open(REPO_ROOT / "pyproject.toml", "rb") as fh:
        cfg = tomllib.load(fh)
    out = set()
    for spec in cfg["project"]["dependencies"]:
        name = spec.split(";")[0]
        for sep in (">=", "<=", "==", "!=", "~=", ">", "<", "["):
            name = name.split(sep)[0]
        out.add(name.strip().lower().replace("_", "-"))
    return out


def _module_level_third_party_imports() -> dict[str, set[str]]:
    """Third-party modules imported at module scope, mapped to source files.

    Includes imports nested directly in a top-level ``try:`` block, since those
    still execute at import time.
    """
    found: dict[str, set[str]] = {}
    for path in PACKAGE.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
            continue

        nodes = []
        for node in tree.body:
            nodes.append(node)
            if isinstance(node, ast.Try):
                nodes.extend(node.body)

        for node in nodes:
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            else:
                continue
            for name in names:
                if name == "fairllms" or name in sys.stdlib_module_names:
                    continue
                found.setdefault(name, set()).add(str(path.relative_to(REPO_ROOT)))
    return found


def test_module_level_imports_are_declared_core_dependencies():
    """Anything imported at module scope must be a core dependency, not an extra."""
    declared = _declared_core_dependencies()
    offenders = {}
    for module, files in _module_level_third_party_imports().items():
        dist = DIST_NAME.get(module, module).lower().replace("_", "-")
        if dist not in declared:
            offenders[dist] = sorted(files)

    assert not offenders, (
        "These are imported at module scope but are not core dependencies, so "
        "`import fairllms` fails on a clean install:\n"
        + "\n".join(
            f"  {dist}\n" + "\n".join(f"    {f}" for f in files)
            for dist, files in sorted(offenders.items())
        )
        + "\n\nEither add them to [project.dependencies] in pyproject.toml, or "
        "move the import inside the function that needs it."
    )


def test_version_is_single_sourced():
    """pyproject must read the version from fairllms/_version.py, not duplicate it."""
    tomllib = pytest.importorskip("tomllib")
    with open(REPO_ROOT / "pyproject.toml", "rb") as fh:
        cfg = tomllib.load(fh)
    project = cfg["project"]
    assert "version" not in project, (
        "version is hardcoded in pyproject.toml; it must stay dynamic so "
        "fairllms/_version.py is the single source of truth"
    )
    assert "version" in project.get("dynamic", [])
    attr = cfg["tool"]["setuptools"]["dynamic"]["version"]["attr"]
    assert attr == "fairllms._version.__version__"


def test_runtime_version_matches_version_module():
    import fairllms
    from fairllms import _version

    assert fairllms.__version__ == _version.__version__


def test_version_helper_script_agrees_with_package():
    """scripts/package_version.py is what the release workflow tags against.

    It must return the real version without importing fairllms. A regression here
    means a release could be tagged with a version that isn't the package's.
    """
    import subprocess

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "package_version.py")],
        capture_output=True,
        text=True,
        check=True,
    )
    from fairllms import _version

    assert result.stdout.strip() == _version.__version__


def test_license_file_exists():
    """pyproject declares MIT; the file it points at must actually be there."""
    tomllib = pytest.importorskip("tomllib")
    with open(REPO_ROOT / "pyproject.toml", "rb") as fh:
        cfg = tomllib.load(fh)
    assert cfg["project"]["license"] == "MIT"
    for rel in cfg["project"]["license-files"]:
        assert (REPO_ROOT / rel).is_file(), f"missing declared license file: {rel}"


def test_sdist_manifest_includes_diagnostic_reproducibility_material():
    """Golden fixtures, evidence guides, and examples ship in the source artifact."""
    golden_root = REPO_ROOT / "tests" / "data" / "golden"
    golden_json = sorted(golden_root.rglob("*.json"))
    assert golden_json, "expected at least one golden JSON fixture"

    manifest = REPO_ROOT / "MANIFEST.in"
    assert manifest.is_file(), "MANIFEST.in is required to declare sdist fixtures"
    rules = [
        shlex.split(line, comments=True)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert ["recursive-include", "tests/data/golden", "*.json"] in rules, (
        "MANIFEST.in must recursively include tests/data/golden/**/*.json so "
        "paper-parity fixtures ship in the source distribution"
    )
    assert ["recursive-include", "docs", "*.md"] in rules
    assert ["recursive-include", "examples", "*.md", "*.py"] in rules
    assert (REPO_ROOT / "docs" / "preparing_audit_evidence.md").is_file()
    assert (REPO_ROOT / "examples" / "scorer_rate_gap_diagnostic.py").is_file()
    assert (REPO_ROOT / "examples" / "scorer_distribution_gap_diagnostic.py").is_file()
    assert (
        REPO_ROOT / "examples" / "scorer_counterfactual_sensitivity_diagnostic.py"
    ).is_file()
    assert (
        golden_root
        / "diagnostics"
        / "score_counterfactual_sensitivity"
        / "bbq_age_toxicity_v1.json"
    ).is_file()
