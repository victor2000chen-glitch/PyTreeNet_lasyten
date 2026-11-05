# Upstream branch integration

This standalone repository is not a Git-history fork of upstream PyTreeNet.
Do not replace the upstream repository with these files. The companion
`pytreenet-upstream-lasyten.patch` adds the experimental `pytreenet_lasyten`
namespace alongside the existing `pytreenet` namespace.

Recorded bases (2026-11-04):

| Patch | Repository | Base commit |
| --- | --- | --- |
| `pytreenet-lasyten-cleanup.patch` | victor2000chen-glitch/pytreenet_lasyten | `a7a520ff454b927cdd2c52c8c8434531294eebfe` |
| `pytreenet-upstream-lasyten.patch` | Drachier/PyTreeNet | `5561eca14e032f1bc106042e2b9457d210fd4269` |

The integration leaves upstream `pytreenet/` source files unchanged. It adds
backend source, tests, examples and documentation. Packaging discovers both
namespaces. The experimental branch requires Python 3.12+, updates the upstream
test workflow accordingly, adds optional MPI/HK dependencies, and corrects the
upstream readme filename used for package builds. This Python requirement is a
branch metadata change; it should be reviewed before upstream acceptance.

## Apply to the uploaded standalone repository

Use a clean working tree and download the cleanup patch first:

```bash
git clone https://github.com/victor2000chen-glitch/pytreenet_lasyten.git
cd pytreenet_lasyten
git switch -c lasyten-backend-cleanup a7a520ff454b927cdd2c52c8c8434531294eebfe
git am /absolute/path/to/pytreenet-lasyten-cleanup.patch
conda activate lasyten_pytreenet
python -m pip install --no-deps --no-build-isolation -e .
python -m pytest -q
git push -u origin lasyten-backend-cleanup
```

## Create a branch based on upstream

```bash
git clone https://github.com/Drachier/PyTreeNet.git PyTreeNet-upstream
cd PyTreeNet-upstream
git switch -c feature/lasyten-backend 5561eca14e032f1bc106042e2b9457d210fd4269
git am /absolute/path/to/pytreenet-upstream-lasyten.patch
python -m pip install --no-deps --no-build-isolation -e .
```

Push to a fork you control, then open a pull request targeting the intended
upstream branch. A branch in the standalone repository alone does not acquire
upstream history. Neither patch changes the installed external LASyTen or
hdf5_io libraries.

## Native checks before a production LRZ run

Use your existing compiled LASyTen environment on an allocated compute node:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest -q -m native
python -m examples.native_symmetry_check
srun --ntasks=4 --cpus-per-task=1 python -m examples.benchmark_workers \
    --executor mpi --backend u1xu1 --nb 3 --chi 8 --warmup 3 --steps 3
```

The symmetry check covers U1xU1 QR, overlaps, TTNO application, both-spin JW
excitations, BUG versus an exact exponential, and DMRG versus exact
diagonalization. Separate tests cover native checkpoint round trips. A failure
in the native external HDF5 loader still requires a matching external-library
fix; compact workers avoid that serialization path but do not repair it.
