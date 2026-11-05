# Cleanup 0.1.1 — 2026-10-28

## Source selection and repository layout

The uploaded directory mixed a Python package with a project root and another
complete package at `pytreenet_lasyten/pytreenet_lasyten/`. The nested copy had
no unique files; seven source files differed, exactly at the installed BUG
optimization hooks. The outer copy additionally contained `bug_performance.py`.
The outer copy is now the single canonical source package.

This matters for execution: the mixed root exposed its `random/` package as a
top-level module, shadowing Python's standard-library `random`. The original
archive produced nine test failures in fresh-process imports and spawned BUG
workers in this environment. Consolidating the layout resolved those failures.
Tests now guard both the import location and standard-library imports.

Removed generated Python 3.14 bytecode, cache directories and the stale nested
source tree from the delivered source layout. Both installer histories remain
in `maintenance/original_installer_backups.zip`, outside the import path. Added
ignore/line-ending rules and explicit package discovery; wheels contain only
the canonical Python package. The source distribution includes tests/examples.

## Code maintenance

- Added missing deferred type imports using `TYPE_CHECKING` where runtime
  imports would create cycles, and repaired missing `Tuple`, `Sequence` and
  `npt.NDArray` annotation names.
- Replaced the private NumPy typing import with `numpy.typing`.
- Removed duplicate `TensorProduct`/`SVDParameters` imports and a few unused
  standard-library imports. Existing public classes, functions and module names
  (including historical spelling) remain available.
- Removed unreachable Davidson similarity-sorting branches. The option was
  already explicitly unsupported and raised at entry; that guard remains.
  Corrected the existing diagnostic string interpolation, including the
  literal `%d` previously printed after a dependent Davidson guess.
- Updated the adapter docstring: native tensor-valued RK23/RK45 and one-site
  Davidson are already integrated.
- Made the import regression test use pytest's temporary directory rather than
  the Unix-only `/tmp` path.

## Verification and documentation

Added a reusable `examples.bug_performance_check` with NumPy/native modes,
real/complex-time checks against dense evolution, and restoration of temporary
performance settings. Included contraction-planner/charged-environment tests
from the performance-installer development work. Documented all feature flags,
profiling semantics, installation layout and memory limitations. Preserved the
October 25 validation report under `docs/` and replaced `VALIDATION.md` with the
results for this cleanup.

## Numerical scope

The contraction planners, native arithmetic helper, local DMRG kernels, RK
coefficients, truncation algorithm, MPI scheduling and serialization protocol
were not redesigned. No topology, physical/Jordan–Wigner ordering, particle
number, tolerance, bond dimension or checkpoint format changes were made.
Both BUG performance helper modules are byte-identical to the uploaded versions.
This is maintenance and packaging work, not an additional speedup claim.
