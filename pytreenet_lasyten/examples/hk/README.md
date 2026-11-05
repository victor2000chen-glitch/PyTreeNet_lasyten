# Two-sector HK example

From the root, with your external `utils.py` accessible:

```bash
export PYTHONPATH="$HOME/DMFT${PYTHONPATH:+:$PYTHONPATH}"
python -m examples.hk.lasyten_HK_mpi --check-imports
srun --ntasks=2 --cpus-per-task=1 \
    python -u -m mpi4py -m examples.hk.lasyten_HK_mpi
```

The current HK model helper and sparse transfer helper live at the repository
root and are source-distribution files. The example retains the supplied nb=5,
three-orbital, chi=15/10, 20-sweep model. Initialization is still on both ranks;
rank zero runs DMRG/excitations. Removal evolves on rank zero, addition on rank
one. No subtree executor is added.

RK45 remains the uploaded default. M=10 is ten intervals plus t=0 (11 snapshots).
Gram matrices are reused, H projection uses <v_i|H v_j>, and physical excitation
weights are applied once. CSV names and headers are unchanged. Full residuals
are for the compressed H action. Native broadcasts carry allowed blocks.

The algebra tests are not compiled HK/DMRG/MPI validation. Run a native pilot
in the working LASyTen environment before production.
