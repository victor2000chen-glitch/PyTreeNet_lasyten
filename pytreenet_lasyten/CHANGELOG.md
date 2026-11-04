# Changelog

## 0.2.0 — 2026-11-04

- Restore the canonical `pytreenet_lasyten/` directory expected by packaging and
  tests; remove 107 tracked Python bytecode files.
- Preserve verified U1/U1xU1 kernels, the actual-QR-parent-space initializer fix,
  contraction planner, native RK arithmetic and compact worker modules.
- Permanently export compact MPIBUG classes; retain explicitly named legacy
  full-tree workers for comparisons, without installer activation markers.
- Add compact-worker regressions and a benchmark entry point.
- Update HK projection/Lehmann helpers and add the matching two-sector example
  and sparse native MPI transfer helper. Preserve the uploaded RK45 default.
- Add installation, symmetry, MPI, validation and upstream-integration guidance
  and NumPy CI. Compiled native validation is explicitly separate.

Earlier validation documents are historical records, not current test results.
