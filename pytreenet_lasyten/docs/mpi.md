# Compact subtree workers

`MPIBUG`/`MPIFixedBUG` permanently export `_mpi_bug_fast` classes. The original
full-tree/HDF5 protocol remains under explicit `LegacyMPIBUG` names.

Tasks contain descendant and boundary-parent tensors, required ancestor/neighbour
metadata for QR, and relevant environments. Native objects are encoded as plain
CPU numerical blocks and leg/charge records. No GPU transport is implemented.

Persistent workers cache eight immutable H fragments. On a miss they request a
fragment and retry. Executors have no guaranteed affinity; inspect misses after
warmup. Scheduling starts from tensor sizes and then uses measured compute time.
Ancestor/root updates and truncation remain on the coordinator.

Ordinary H tensor replacements are tracked. After **in-place coefficient/block
edits**, call `evolution.invalidate_worker_hamiltonian()` or disable H caching.
Time-dependent configurations bypass caching. Keep `config.subtree_workers=1`;
the MPI allocation controls process count.

`last_worker_stats` reports bytes, cache misses, encode/decode, compute, initial
cache, coordinator completion, truncation and whole-step timings. Nested times
overlap. Worker `process_peak_rss_mib` is a lifetime peak, not simultaneous
aggregate RSS; sum simultaneous process samples for memory comparisons.

Start with one coordinator plus three workers, one BLAS thread each, and several
warmups. Compare fidelity, norm and phase. `--include-legacy` requires the old
native HDF5 codec to work. The two-sector HK example uses serial BUG per sector;
particle/hole MPI and subtree MPI are separate layers.
