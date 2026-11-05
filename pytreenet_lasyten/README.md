# PyTreeNet with LASyTen kernels

Version **0.1.1** consolidates the uploaded development tree into one installable
package. The native BUG speedups and contraction planner are already integrated
and load automatically in fresh processes. Do not rerun a patch installer.

The project root contains `pyproject.toml`, `tests/`, `examples/`, and one
`pytreenet_lasyten/` source directory. Run commands from the project root, not
from inside the Python package. The uploaded outer copy was retained because
it contains the installed BUG improvements; the older duplicate was removed.
See `CHANGELOG.md` for the complete cleanup scope.

## Install

Extract this archive into a **new directory** named `pytreenet_lasyten_clean`.
Do not overlay it on the previous mixed package/project directory. From the
directory containing `pyproject.toml`, activate your existing environment:

```bash
conda activate lasyten
python -m pip install --no-deps --no-build-isolation -e .
python -c "import pytreenet_lasyten; print(pytreenet_lasyten.__file__)"
```

`--no-deps` assumes you installed the NumPy, SciPy, tqdm and h5py dependencies
in that environment. Otherwise use `python -m pip install -e .`.
The distribution name is `pytreenet-lasyten`; the Python import name is
`pytreenet_lasyten`. The old `pytreenet` package is not a dependency. Confirm that the printed path
points into this new checkout, then restart Jupyter kernels and MPI workers.
Your existing imports, including
`from pytreenet_lasyten.dmrg.dmrg_sparse import DMRGAlgorithm_sparse`, remain valid.
An editable installation makes future source edits visible after restarting
the importing process; it does not replace objects already loaded in a kernel.

For native tensors, first install your matching LASyTen source and its
`external/hdf5_io` dependency in the same environment. The LASyTen snapshot
used for this work requires Python >=3.14; the NumPy-only package/tests can run
on Python >=3.12. LASyTen remains a separately compiled dependency. Keep its
working version when moving this package to LRZ.

For MPI, build mpi4py against the loaded LRZ MPI module. Native worker transport
also requires h5py >=3.13 and the checkpoint-capable LASyTen/hdf5_io build on
every rank. Verify these dependencies on the target machine before running. Optional extras
are `test`, `dev`, `mpi`, `notebook` and `sim-utils` (the existing
experiment-notification helpers). `dev` adds pytest, Ruff and build tooling.

## Scope of native support

The migrated path includes tensor storage, U(1) state/TTNO construction, direct
TTNO application, overlaps and sandwich contractions, excitations, native
checkpoints, one-site sparse DMRG, and adaptive BUG. Other inherited algorithms
still contain NumPy kernels. Importability does not imply every legacy
algorithm accepts a native tensor.

Public QR/SVD wrappers now forward modes explicitly. This LASyTen backend
supports reduced exact decompositions. It rejects unsupported FULL/KEEP bond
embeddings instead of silently ignoring the request. Native adaptive BUG and
DMRG canonicalization explicitly use reduced QR, matching the effective
behavior of the working snapshot. Native fixed-rank BUG is limited to sector
dimensions for which its requested KEEP decomposition is supported.

Native two-site sparse DMRG now raises before mutating the state. Use
`site="one-site"`. Sparse DMRG always uses the adapted Davidson implementation,
including when PySCF happens to be installed.

## MPI integration

`pytreenet_lasyten.time_evolution.mpi_bug` now provides `MPIBUG` and
`MPIFixedBUG`. It uses the same preparation and finishing kernels as serial
BUG. The coordinator prepares ancestor gauges, workers update disjoint
subtrees, and the coordinator merges results and performs truncation.

NumPy-only task messages are frozen with pickle before submission. Native
messages use LASyTen's HDF5 tensor serializer, retaining blocks, spaces,
labels and hidden charge state. Node space objects are not pickled and native
tensors are not converted to dense arrays. Small config objects and solver
options must remain picklable.

**Memory/communication limit:** each frontier task currently sends full state,
parent-state and Hamiltonian snapshots, then returns its updated subtree and
boundary tensors. These serialized snapshots consume additional memory. This
is task distribution, not the separate streaming/distributed-storage MPI
implementation. It does not guarantee memory proportional only to subtree
size. Start with a small worker count and measure process and total RSS.

All ranks must enter the executor context. Only the coordinator constructs and
runs the integrator:

```python
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor
from pytreenet_lasyten.time_evolution.bug import BUGConfig
from pytreenet_lasyten.time_evolution.mpi_bug import MPIBUG
from pytreenet_lasyten.time_evolution.time_evolution import TimeEvoMode

with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
    if executor is not None:
        # Construct/load psi and H here using pytreenet_lasyten classes.
        config = BUGConfig(max_bond_dim=16, rel_tol=1e-10, total_tol=1e-12,
                           time_evo_mode=TimeEvoMode.RK23, subtree_workers=1)
        evolution = MPIBUG(psi, H, 0.01, 0.05, {}, config=config,
                           solver_options={"rtol": 1e-8, "atol": 1e-10},
                           executor=executor, subtree_tasks=3)
        evolution.run()
```

Use one process per worker; keep `subtree_workers=1`. `subtree_tasks` selects
the disjoint frontier, whose actual size depends on tree branching. The
coordinator is an additional MPI rank. Reuse the same executor across steps.

## Verification

Set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` and `MKL_NUM_THREADS=1` for
the initial comparisons. From the extracted project directory:

```bash
python -m pytest -q
python -m examples.bug_performance_check --backend numpy
python -m examples.bug_performance_check --backend lasyten
python -m examples.mpi_bug_smoke --executor process --verify
mpiexec -n 3 python -m examples.mpi_bug_smoke --backend numpy --verify
mpiexec -n 3 python -m examples.mpi_bug_smoke --backend lasyten --verify
python -m pytest -q -m native
```

`bug_performance_check` compares optimized and previous contractions over two
steps against an independent dense exponential for RK23/RK45 and real/complex
time. The native version requires compiled LASyTen and fails if it is missing;
it does not silently fall back to NumPy. It temporarily enables lazy node copies
to test that opt-in path, and restores environment/settings afterward.

The MPI executable uses a tiny three-site number-conserving Hamiltonian. `--verify`
compares each distributed step with serial BUG using normalized squared
fidelity, norm and relative-phase agreement. It does not construct your HK impurity problem.
`--fixed-rank` selects the fixed-rank algorithm.

`VALIDATION.md` records which checks actually ran while preparing this archive.
The compiled native path must be checked on LRZ before production use.

## Performance controls

See [docs/bug_performance.md](docs/bug_performance.md) for feature switches,
profiling and memory limitations. Numerical tolerances, charge sectors, bond
caps, physical site order and tensor topology were not changed by this cleanup.

## Maintenance

```bash
python -m ruff check pytreenet_lasyten tests examples
python -m build --no-isolation
```

A source checkout contains examples and tests; the wheel contains only the
importable package. The historical installer backups are preserved as one
non-importable ZIP under `maintenance/`; they are not included in wheels.
The upstream license and Davidson/PySCF attribution are retained. Checkpoint
format identifiers remain `pytreenet.native_ttn` for existing-file compatibility.
