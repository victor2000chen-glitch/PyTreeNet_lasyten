# PyTreeNet with LASyTen

`pytreenet_lasyten` adds experimental block-sparse tensor support to
[PyTreeNet](https://github.com/Drachier/PyTreeNet) through the compiled LASyTen
backend. It is installed separately from the NumPy-based `pytreenet` package.

- **Symmetries:** `U(1)` conserves total particle number; `U(1) × U(1)` conserves the up- and down-spin particle numbers separately.
- **Tensor operations:** construct tree tensor network states and operators, compute overlaps, apply operators directly, and create particle and hole excitations.
- **Algorithms:** sparse one-site and two-site DMRG and adaptive BUG time evolution, with optional MPI subtree workers.
- **Complex Time impurity solver:** Helper functions for the Complex Time Krylov subspace (CTKS) expansion such as Krylov vector generation and projection are found in Complex_bug_impurity_solver_lasyten.py

## Install

From the **repository root**, activate an environment with a compatible LASyTen
build, then install the backend subproject:

```bash
conda activate lasyten_pytreenet
cd pytreenet_lasyten
python -m pip install -e .
python -c 'import lasyten, pytreenet_lasyten; print(pytreenet_lasyten.__file__)'
```

LASyTen is installed separately; the development build used here uses Python
3.14. For MPI runs, use `mpi4py` matched to your cluster's MPI runtime. Native
HDF5 checkpoints require a compatible LASyTen HDF5 build and `hdf5_io`.

## Verify and explore

From the same directory, run the native smoke test:

```bash
python -m examples.native_symmetry_check
```

See [symmetry support](docs/symmetries.md), [MPI BUG](docs/mpi.md), and the
[Hubbard–Kanamori example](examples/hk/README.md) for more details. Licensed
under the [EUPL 1.2](LICENSE).
