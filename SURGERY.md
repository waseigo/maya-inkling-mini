# Maya Mini on supra — surgery v1 (no host expert pin)

Target: 2×16 GB Blackwell, 16 GB DDR5. Experts live on the GPUs. Host keeps the process only.

Do **not** enable dual-GPU `g_fast` CUDA graphs. Do **not** lift `--draft: one GPU only`. Do **not** touch `IK2_DENSE` die. Do **not** reformat the file.

Write-set: `src/ik2/ik2_main.cu` only.

## Changes

### 1. Default mmap on dual-GPU (around line 5097)

Today:

```c
const bool host_mmap = getenv("IK2_HOST") ? std::string(getenv("IK2_HOST")) == "mmap" : g_fast_want;
```

Wanted: mmap also when `dual()` is true, unless `IK2_HOST=pinned`.

```c
const bool host_mmap = getenv("IK2_HOST")
    ? std::string(getenv("IK2_HOST")) == "mmap"
    : (g_fast_want || dual());
```

`dual()` is already set by this point (`g_ep` / `g_tp` around 4675).

### 2. Do not `MADV_WILLNEED` the whole expert set (around 5104–5111)

The Linux `madvise(..., MADV_WILLNEED)` on every expert tensor pulls ~14.7 GiB into page cache on a 16 GB host.

- Keep the Windows prefetch as-is.
- On Linux: skip WILLNEED when `n_gpu > 1`. Single-GPU Mini on a 24 GB card can keep it.

### 3. After EP/TP cache warm-fill, `MADV_DONTNEED` the mapped expert ranges

The warm fill (search `if ((wf && *wf) || g_ep)`) copies owned experts into each GPU cache. Right after that `cudaStreamSynchronize` + the `warm: %d experts` printf, if `host_mmap && n_gpu > 1`, walk the same mapped ranges used in the mmap branch and `madvise(..., MADV_DONTNEED)` them.

Reuse the `exp_jobs` / `S.data(J.name)` addresses already used to set `Ly.host`. Do not invent a second mapping.

If a helper that records those `(ptr,len)` pairs at mmap time is cleaner than walking `exp_jobs` twice, do that — keep it local to this file, no new headers.

### 4. Dual-GPU expert-cache room (around 5462)

Today 2 GiB is reserved unless `g_fast_want`. Dual mmap needs the 256 MiB resident-style hole so both 16 GB cards can hold every owned expert.

```c
const size_t room = getenv("IK2_ROOM_MIB") ? (size_t) std::max(0, atoi(getenv("IK2_ROOM_MIB"))) << 20
                    : (g_fast_want || dual()) ? (size_t) 256 << 20
                                              : (size_t) 2 << 30;
```

## Out of scope

- `g_fast` / `g_fast_want` for `n_gpu != 1`
- `--draft` on two GPUs
- `IK2_DENSE` with `--gpus > 1`
- Python server, CMake, docs, configs
- Compile or run

## Done when

`rg` shows the four behaviors above, the file still compiles in your head (brackets/ifdefs intact), and a 10-line `SURGERY_DONE.md` lists the line numbers you changed.
