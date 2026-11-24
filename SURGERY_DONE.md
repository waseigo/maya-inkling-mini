# SURGERY_DONE — surgery v1 (no host expert pin)

Repo: maya-inkling-mini @ fcd523f, write-set `src/ik2/ik2_main.cu` only (16 insertions, 4 deletions, no reformat).

## Changed lines (final file)

| # | Change | Lines |
|---|--------|-------|
| 1 | `host_mmap` defaults to mmap when `dual()`; `IK2_HOST` still forces pinned/mmap explicitly | 5097–5099 |
| 2 | Linux `MADV_WILLNEED` skipped when `n_gpu > 1` (Windows prefetch untouched) | 5113 |
| 3 | After EP/TP warm-fill, `MADV_DONTNEED` the mapped expert ranges (Linux, `host_mmap && n_gpu > 1`), walking the same `exp_jobs` / `S.data(J.name)` ranges used in the mmap branch | 5526–5535 |
| 4 | `IK2_ROOM` defaults to 256 MiB when `dual()` as well as `g_fast_want` | 5464–5466 |

## Summary (5 lines)

1. Dual-GPU Mini now mmaps the expert files by default instead of pinning ~14.7 GiB of host memory; `IK2_HOST=pinned` still overrides.
2. The read-ahead `MADV_WILLNEED` no longer pulls the whole expert set into the page cache on multi-GPU hosts, where 16 GB of RAM cannot hold it.
3. Once the warm fill has copied the owned experts into both GPUs' caches, the mapped ranges are `MADV_DONTNEED`'d so the host pages drop back to disk (they re-fault on demand, nothing is unmapped).
4. The expert-cache reserve shrinks from 2 GiB to 256 MiB on dual-GPU hosts, so both 16 GB cards can hold every owned expert.
5. Nothing else changed: no dual-GPU `g_fast`/graphs, `--draft` still one GPU, `IK2_DENSE` die untouched, no other files edited, no compile run, nothing committed.
