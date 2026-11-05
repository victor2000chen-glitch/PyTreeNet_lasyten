# Historical installer backups

`original_installer_backups.zip` preserves both original rollback directories
from the uploaded outer package, including their manifests and byte-exact files:

- `_lasyten_bug_speedups_backup/`
- `_bug_performance_v2_backup/`

They are kept outside the importable package so stale numerical modules cannot
be mistaken for active code or included in a wheel. The removed nested package
had no unique files, and its backup copies were identical to these originals.

The cleaned sources already contain both sets of optimizations. Do not run the
old installer or its rollback against this checkout: its manifests refer to
specific historical file hashes and locations. The archived files are retained
for comparison or manual recovery in a separate disposable checkout. Use the
performance switches documented in `../docs/bug_performance.md` for comparisons.
