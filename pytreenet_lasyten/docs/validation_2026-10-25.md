# Validation of this cleaned snapshot

Prepared from the uploaded `pytreenet_lasyten.zip` on 2026-10-25.

## Checks completed here

- **108 regression checks passed; two compiled-LASyTen checks skipped.**
- A freshly built wheel installed into an isolated directory. All **174 package
  modules** imported with imports of the old `pytreenet` package explicitly
  blocked. The validation environment included optional plotting dependencies.
- Sparse DMRG selected the adapted Davidson even with a fake installed PySCF
  solver present, and its NumPy one-site ground energy matched exact
  diagonalization on a small complex-tensor problem.
- QR/SVD reconstruction and mode dimensions passed for rectangular and
  permuted complex tensors. Protected native inputs verified mode forwarding
  without implicit NumPy conversion.
- Serial BUG matched an independent dense matrix exponential, including the
  temporary-parent contraction cases and recursive solver-option forwarding.
- Adaptive and fixed-rank worker propagation matched serial complex state
  vectors over two full steps, using spawned processes and requested frontier
  sizes 1, 2, 3 and 8. Original inputs remained unchanged. Worker errors did
  not replace the coordinator's state.
- A two-step spawned-process smoke run passed fidelity, norm and relative-phase
  agreement against serial propagation.
- The native transport protocol round-tripped nonpicklable tensor/space
  doubles, labels, hidden state and lazy leg permutations through in-memory
  HDF5 using the supplied **Python reference hdf5_io implementation**. The
  doubles reject pickle and implicit array conversion. This validates the
  transport schema, not the compiled LASyTen serializer.
- Protected U(1) kernel models checked the actual shipped prepared-action hook,
  charge-compatible contractions, fused arithmetic/conjugation, size guards,
  input immutability and RK23/RK45 evolution against independent dense
  operators. These models are correctness checks, not performance measurements.
- The profile-guided speedup helper is byte-identical to the uploaded installed
  helper and the latest standalone installer. Reinstallation was idempotent.
  Rollback was tested on a disposable copy and retained the consistency fixes.

The checks ran with Python 3.12. The native LASyTen source used in your working
environment requires Python >=3.14 and was not compiled in this environment.

## Checks to run on LRZ

The real `mpiexec` attempt here stopped in MPI initialization: the execution
environment denied UCX socket/IPC operations. It did not reach the BUG worker
code. No successful Intel MPI run or native C++ BUG run is claimed here.

After installing your matching LASyTen/hdf5_io builds and LRZ mpi4py:

```bash
python -m pytest -q
mpiexec -n 3 python examples/mpi_bug_smoke.py --backend numpy --verify
mpiexec -n 3 python examples/mpi_bug_smoke.py --backend lasyten --verify
python -m pytest -q -m native
```

The two compiled native tests exercise TTNS/TTNO checkpoint round trips,
native spawned-worker propagation, serial/native agreement, and excitation
creation with the charge centre initially away from the root. The extra
transport-protocol tests use your installed hdf5_io when it is available.

Before a large bath job, compare serial and MPI propagation of the same saved
HK input, including norm, relative phase, energy and fidelity, and measure
aggregate memory across all ranks. Full task snapshots and HDF5 message buffers
are additional allocations; this scheduler is not the separate streaming MPI
implementation.
