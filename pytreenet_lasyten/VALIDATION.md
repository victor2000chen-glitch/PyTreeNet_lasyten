# Repository audit and validation — 2026-11-04

## Scope and provenance

Audited upload: `a7a520ff454b927cdd2c52c8c8434531294eebfe` from
[victor2000chen-glitch/pytreenet_lasyten](https://github.com/victor2000chen-glitch/pytreenet_lasyten).
The companion integration branch uses PyTreeNet base
`5561eca14e032f1bc106042e2b9457d210fd4269`.

“Current” means matching the latest patch artifacts available from our work,
not certification against every external LASyTen release. Component hashes
are recorded in [docs/audit-manifest.json](docs/audit-manifest.json).

| Component | Audit finding |
| --- | --- |
| Single U1 state, TTNO and excitation helpers | Match the current U1xU1 integration payload byte for byte |
| U1xU1 charges, initializer, TTNO, excitations and HK construction | Match the latest payload, including the actual-QR-parent-space fix |
| LASyTen tensor adapter | Algorithm definitions match; only documentation/whitespace differ |
| BUG contraction/copy performance module | Matches current performance payload; source hooks are active |
| Compact MPI workers and sparse-block wire | Match the latest subtree-worker payload |
| HK projection/propagation/Lehmann helper | Four older functions updated from the latest matching helper |
| Two-sector HK driver and direct rank transport | Included with the matching updated helper |

The four updated HK functions are `pivoted_cholesky`, `krylov_projection_TTN`,
`Complex_time_evolution`, and `lehmann_sector_TTN`. Other helper functions were
preserved. No U1/U1xU1 numerical kernel was replaced during repository cleanup.

## Repository repairs

The upload placed package contents directly at the Git root, while packaging
and tests expected `pytreenet_lasyten/`. This also exposed `random/` at the top
level, where it could shadow Python's standard library. Source files now live
in one canonical package directory. Removed 107 tracked bytecode files.

`MPIBUG` and `MPIFixedBUG` explicitly export the compact implementations. The
previous classes remain available as `LegacyMPIBUG` and `LegacyMPIFixedBUG`.
No performance installer is needed: the source hooks remain active.

Added packaging exclusions, source-distribution support files, CI, native
regression checks, symmetry/MPI docs, and integration notes. Historical
validation reports remain under `docs/validation_2026-10-*.md`.

## Executed checks

Environment: Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, pytest 9.1.1.
The real MPI test used Open MPI 5.0.7 and mpi4py.

| Check | Result |
| --- | --- |
| Standalone regression suite | **165 passed, 18 skipped; 16 subtests passed** |
| LASyTen tests inside upstream branch | **165 passed, 18 skipped; 16 subtests passed** |
| Editable installation | Builds/installs as `pytreenet-lasyten` 0.2.0 |
| Wheel/source distribution | Both build; expected support files included; no bytecode in wheel |
| Isolated wheel imports | All 182 discovered modules import with old `pytreenet` blocked |
| Compact public exports | Both classes resolve to `_mpi_bug_fast` |
| Python 3.13 syntax compilation | Package, tests, examples and HK support files compile |
| Four-rank MPI, three NumPy BUG steps after warmup | Fidelity **1.0**, phase about **5.6e-17 rad**, norm squared about **0.9999999999999991** |
| Upstream branch wheel | Builds with both namespaces |
| Upstream `pytreenet/` source | Unchanged relative to the recorded upstream base |

The small MPI run validates execution and agreement, not large-native speedup.
Runtime depends on branch sizes, solver tolerances, worker overhead, and CPU
and memory allocations.

## Native limits and LRZ checks

Compiled LASyTen and hdf5_io are **not installed here**. Skipped tests and API
test doubles do not establish compiled-native correctness. Python 3.14
execution and the GitHub CI matrix have not been run here.

Shipped native checks cover nine U1xU1 initializer cases (`nb=5,7,11`,
`chi=8,16,50`), plus exact four-mode QR/overlap, both-spin JW excitation,
TTNO application, DMRG and BUG checks, and existing worker/checkpoint tests.
Run in the compiled LRZ environment:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest -q -m native
python -m examples.native_symmetry_check
srun --ntasks=4 --cpus-per-task=1 python -m examples.benchmark_workers \
    --executor mpi --backend u1xu1 --nb 3 --chi 8 --steps 3 --warmup 3
```

Compact workers transport sparse CPU blocks. `save_ttn`/`load_ttn` and the
legacy codec still require a compatible external LASyTen/hdf5_io build. This
repository does not include or certify external C++ HDF5 loader fixes.
The full HK example also requires the user's `utils.Discretisation_function`;
its bath discretization was not fabricated or replaced.

This is an experimental CPU backend. Native two-site DMRG and unsupported
charge embeddings remain explicitly rejected. Other inherited algorithms are
not automatically native-compatible. GPU support is not added by this cleanup.

## Upstream baseline failures

The integration does not claim a green full upstream suite. An initial run
stopped after five failures (1224 passed); one was a new test-relative-import
issue and was fixed. Four numerical/test failures were then reproduced
individually on the **unmodified upstream base**:

- `tests/contractions/test_state_operator_contraction.py::TestMatrixElement::test_standard`
- `tests/core/truncation/test_variational.py::TestVariationalFittingSmall::test_var_fitting_lower_bd`
- Both one-step cases in `tests/time_evolution/test_magnus.py::TestMagnus`

A subsequent complete integration-suite attempt exceeded a 120-second limit;
a concurrent full baseline attempt was terminated with exit 137. These are
not complete test runs. The four-case baseline reproduction completed and
failed all four cases. The standalone/backend suite completed successfully.
Upstream numerical source/tests were not modified to hide these failures.

The original upstream CI job is retained with Python 3.12. The added LASyTen
CI job targets the new tests and an independent four-rank NumPy MPI smoke
check. Native LRZ checks remain a separate prerequisite for production runs.

The uploaded GitHub main advanced to `a7a520f` during the audit. Its only
change was the shortened HK module docstring, which has been preserved.
