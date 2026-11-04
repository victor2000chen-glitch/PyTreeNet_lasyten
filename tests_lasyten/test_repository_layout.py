"""Prevent the duplicate-package and stdlib-shadowing regression in the upload."""

from pathlib import Path
import os
import subprocess
import sys

import pytreenet_lasyten

PROJECT = Path(__file__).resolve().parents[1]


def test_canonical_source_is_the_only_importable_copy():
    package = PROJECT / "pytreenet_lasyten"
    assert Path(pytreenet_lasyten.__file__).resolve().parent == package
    assert not (PROJECT / "__init__.py").exists()
    assert not (PROJECT / "random").exists()
    assert not (package / "pytreenet_lasyten").exists()
    assert not list(package.glob("*backup*"))


def test_fresh_process_imports_stdlib_and_optimized_package():
    script = """
import random, secrets, tempfile
from pathlib import Path
import pytreenet_lasyten
from pytreenet_lasyten.util import bug_performance
from pytreenet_lasyten.time_evolution import BUG, MPIBUG
assert hasattr(random, "SystemRandom")
assert Path(pytreenet_lasyten.__file__).resolve().parent == Path.cwd() / "pytreenet_lasyten"
assert bug_performance.plan_chain is not None
"""
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
