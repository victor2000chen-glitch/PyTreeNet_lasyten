# BUG performance controls

The uploaded installed implementation is the baseline for this cleanup. Both
`util/bug_native_speedups.py` and `util/bug_performance.py` are byte-identical to
that input. These switches can compare the installed paths without reinstalling
or modifying the repository. Settings are process-local; pass environment
variables to every worker when testing MPI.

| Variable | Default | Effect |
| --- | --- | --- |
| `PYTREENET_BUG_PLANNED_ACTIONS` | `1` | Plan connected contraction chains anchored on the trial tensor for large native local actions. |
| `PYTREENET_BUG_PLANNED_ENVIRONMENTS` | `1` | Plan native environment contractions jointly. |
| `PYTREENET_BUG_CACHE_PATHS` | `1` | Reuse traversal paths for unchanged ordered topology. |
| `PYTREENET_BUG_SHARE_MPI_ENCODING` | `1` | Reuse identical native task serialization within one root update. |
| `PYTREENET_BUG_LAZY_NODE_COPIES` | `0` | Opt in to copying node metadata on access. |
| `PYTREENET_BUG_PROFILE` | `0` | Collect local solve and aggregate BUG timings. |
| `PYTREENET_BUG_PATH_MIB` | `256` | Dense-equivalent intermediate-size target used to select a contraction path. |

Use `0`, `false` or `off` to disable a boolean switch. The planner target must
be finite and positive. It is **not an RSS limit**: if no available chain fits,
the minimum-peak chain is used and `plans_over_target` is incremented. Shape
estimates are dense-equivalent, not measured block memory or block FLOPs. These
optimizations do not slice contractions or alter truncation tolerances.

```python
import os
from pprint import pprint
from pytreenet_lasyten.util import bug_performance as perf

os.environ["PYTREENET_BUG_PROFILE"] = "1"
perf.reset_stats()
evolution.run_one_time_step()  # Advances your existing integrator.
pprint(perf.stats())
```

`counts` reports planner and serialization use. `seconds` separates recursive
update from truncation. `local_solves` reports calls, elapsed time and native RK
function evaluations/accepted/rejected steps for each node. Results are local
to each process, not an MPI reduction. The aggregate recursive time includes
its local solve times, so do not add those columns together.

`reset_stats()` clears counters and timing rows, not cached plans/traversal
paths. `plan_cache` statistics describe the process lifetime. Benchmark warmed
runs from identical initial states; report thread counts, bond dimensions,
solver tolerances and total memory across ranks. Tiny smoke tests establish
correctness only, not a production speedup.

The MPI scheduler still transmits full state/operator snapshots to workers.
Sharing serialization avoids repeated encoding on the coordinator but does
not reduce the bytes sent to each worker or create distributed tensor storage.

The existing native helper also supports `bug_native_speedups.configure(...)`.
Its settings and counters are separate from `bug_performance`. Dense-equivalent
thresholds on the precontracted small operator do not bound all fallback paths.
