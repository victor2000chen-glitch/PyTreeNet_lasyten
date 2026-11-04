# PyTreeNet LASyTen backend

Experimental LASyTen support for tree tensor networks in the separate
`pytreenet_lasyten` namespace. When added to upstream PyTreeNet, it coexists
with the existing NumPy-based `pytreenet` namespace.

The current source includes bosonic U1 and spin-resolved U1xU1 state/operator
construction, DIRECT TTNO application, overlaps and sandwich contractions,
charge-changing impurity excitations, one-site sparse DMRG, adaptive BUG and
compact MPI subtree workers. Fermion signs come from explicit model JW strings.
Other inherited algorithms have not all been ported to native tensors.

## Install

Run from the root containing `pyproject.toml` and `pytreenet_lasyten/`:

```bash
conda activate lasyten_pytreenet
python -m pip install --no-deps --no-build-isolation -e .
python -c "import pytreenet_lasyten; print(pytreenet_lasyten.__file__)"
```

`--no-deps` assumes the declared dependencies are installed. Otherwise use
`python -m pip install -e '.[test]'`. Python 3.12+ is required for the standalone
package. Your working LASyTen snapshot requires Python 3.14; keep that separately
compiled installation on LRZ. This project does not vendor/rebuild LASyTen.

MPI needs mpi4py built against the cluster MPI runtime (`.[mpi]`). The HK helper
needs `.[hk]`. HDF5 checkpoints and the legacy worker codec need matching
LASyTen/hdf5_io support. **Current compact workers do not use HDF5 transport.**
Restart Python/Jupyter/MPI processes after source changes; no performance patch
installer is needed for this organized checkout.

## Symmetry support

```python
from pytreenet_lasyten.util.hk_u1_model import build_u1_t3ns
from pytreenet_lasyten.util.abelian_charges import fixed_charge

state = build_u1_t3ns(
    norb=3, nb=5, virtual_bond_dimension=16, bath_bond_dimension=16,
    symmetry="u1xu1", total_particles_by_spin=(9, 9), seed=11,
)
print(fixed_charge(state))
```

U1 fixes total N; U1xU1 fixes `(N_up,N_down)`. Caps sum dimensions over all
sectors. For U1 choose `symmetry="u1", total_particles=18` and omit the spin
pair. See [symmetries](docs/symmetries.md).

Native two-site DMRG and unsupported FULL/KEEP charge embeddings are rejected
explicitly. Use native one-site DMRG and DIRECT application. Importability of
other algorithms does not establish native support.

## MPI BUG

```python
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor
from pytreenet_lasyten.time_evolution.bug import BUGConfig
from pytreenet_lasyten.time_evolution.mpi_bug import MPIBUG
from pytreenet_lasyten.time_evolution.time_evolution import TimeEvoMode

with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
    if executor is not None:
        # Construct/load psi and H here.
        config = BUGConfig(max_bond_dim=16, time_evo_mode=TimeEvoMode.RK45,
                           rel_tol=1e-10, total_tol=1e-12, subtree_workers=1)
        evolution = MPIBUG(psi, H, .01, .05, {}, config=config,
                           solver_options={"rtol": 1e-8, "atol": 1e-10},
                           executor=executor, subtree_tasks=3)
        evolution.run()
        print(evolution.last_worker_stats)
```

Every rank enters the persistent executor. One coordinator retains the tree;
workers update disjoint subtrees using scoped state/environment messages and
cached immutable H fragments. Native tensors travel as allowed CPU blocks and
space records; no native object is pickled or densely exported. Ancestor/root
completion and truncation remain on the coordinator. This is not fully
distributed storage. See [MPI details](docs/mpi.md).

## Verify

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest -q
python -m examples.bug_performance_check --backend numpy
python -m examples.benchmark_workers --executor process --backend numpy
mpiexec -n 4 python -m examples.benchmark_workers --executor mpi --backend numpy
```

With compiled LASyTen:

```bash
python -m pytest -q -m native
python -m examples.bug_performance_check --backend lasyten
mpiexec -n 4 python -m examples.benchmark_workers --executor mpi --backend u1xu1 \
    --nb 3 --chi 8 --warmup 3 --steps 3
```

NumPy/test-double results are distinct from compiled native validation.
[VALIDATION.md](VALIDATION.md) records current results and outstanding checks.
See [the HK example](examples/hk/README.md) for the two-sector driver.

## Contributing

The package/tests/examples/docs layout follows PyTreeNet. This is an experimental
additional namespace, not a replacement of upstream `pytreenet`. A companion
integration patch targets a recorded upstream commit; see
[integration notes](docs/upstream-integration.md).

Retain EUPL-1.2 and the existing Davidson/PySCF attribution. Source distributions
include tests/examples/docs; wheels contain the package only.
