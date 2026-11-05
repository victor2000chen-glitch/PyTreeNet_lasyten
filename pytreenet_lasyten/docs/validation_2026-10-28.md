# Validation — cleanup 0.1.1 (2026-10-28)

Validated against the uploaded `pytreenet_lasyten (2).zip`. The optimized outer
package was used as the source; the stale nested package was discarded.

## Completed checks

| Check | Result |
| --- | --- |
| Full regression suite | **151 passed, 8 skipped** |
| Ruff correctness rules (`E9`, `F63`, `F7`, `F82`) | Passed |
| Syntax compilation of package, tests and examples | Passed |
| Wheel build and isolated installation | Passed; version 0.1.1 |
| Import all package modules from isolated wheel | **176 imported**, with old `pytreenet` imports blocked |
| Standalone optimized/previous BUG check | RK23 and RK45, real and complex time; all four NumPy cases passed |
| Two-step spawned-process BUG smoke | Passed; normalized fidelity 1 at both steps, with norm and phase checks |
| Performance helper preservation | Both helper modules byte-identical to uploaded installed versions |

The full suite includes a one-site DMRG/exact-energy comparison, independent
dense exponential checks, charged-environment models, caller-input immutability,
planner permutations, serialization reuse, and adaptive/fixed-rank spawned
worker comparisons for several subtree targets. The planner/charged test doubles
reject unintended dense export; they are not a compiled LASyTen substitute.

The original mixed directory layout produced 9 failures, 97 passes and 4 skips
here. Those failures were fresh-process/standard-library import failures, not
evidence of a numerical error in the original BUG contractions. The cleanup
added tests and repaired the layout; the old validation report is retained at
`docs/validation_2026-10-25.md` as historical information.

Environment: Python 3.12.14, NumPy 2.3.5,
SciPy 1.17.0, Linux. Optional plotting/experiment dependencies were
available for the all-module import check. No production HK timing or memory
benchmark was performed.

## Not run in this environment

Six compiled LASyTen checks and two HDF5 transport checks were skipped because
`lasyten` and `hdf5_io` are unavailable. Actual MPI-rank execution was not run;
the worker checks use `ProcessPoolExecutor` with spawn. Windows and Intel MPI
were not exercised here. Your successful native check before this cleanup is
useful evidence for the preserved kernels, but is not a new run of this archive.

## Check in your existing LASyTen environment

From the new project root, after the editable installation described in README:

```bash
python -m pytest -q
python -m examples.bug_performance_check --backend lasyten
python -m examples.mpi_bug_smoke --backend lasyten --executor process --verify
mpiexec -n 3 python -m examples.mpi_bug_smoke --backend lasyten --verify
```

The explicit native example fails when LASyTen cannot import instead of reporting
NumPy success. Start with one BLAS thread per rank for correctness comparisons;
thread tuning and production speedups require separate measurements.
