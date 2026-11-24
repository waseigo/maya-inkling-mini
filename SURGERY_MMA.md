# Dual-GPU long-ctx attention: MMA + flash (not attn_multi)

**Status:** §3a+§3b **done 2026-11-23.** MMA in `run_layer_attn`; dual defaults `g_group_attn=false` and `IK2_KV=fp8`.

The 23 tok/s “4 GiB leak” was **FP16 KV**: a bench env dropped `IK2_KV=fp8`. Dual now forces fp8 unless the operator sets `IK2_KV`.

| bench (fp8, MMA, split 0.65) | tg | verify |
|---|---|---|
| empty 262k | **101** | 31 ms, 0 misses, GPU0 8.7 GiB free |
| 8k unique filled | **100** | 28 ms |
| **32k unique filled** | **89** | 29 ms — beats flash 81 and IQ4 59 |
| code prompt | still `1160,7,37777,...` | |

**Parent session:** local-ai Grok Build, Maya clone `~/github/maya-inkling-mini` (shallow `fcd523f` + **uncommitted** dual-GPU work in `src/ik2/ik2_{main,kernels}.{cu,cuh}`). Binary on supra: `/home/tisaak/github/maya-inkling-mini/build/maya-ik2` (sm_120, built on turbo).

**Do this yourself.** Do not spawn subagents unless Isaak says so. Compile on **turbo**, rsync to **supra**, bench there. One llama-swap at a time; `GET http://192.168.88.14:8935/unload` before GPU jobs.

---

## 0. Why

DSpark **verify** is `T=2..8` through `run_layer_attn` (`ik2_main.cu` ~1522, attention dispatch ~1705). On dual that uses **`attn_multi_fp8`**, which is ALU-bound (`P ∝ ctx/64`). The 3090 single-GPU path uses tensor-core **`attn_decode_mma` / `attn_block_fp8`**, wired only in `fast_token` / `fast_block` (`g_attn_mma` at 2511). Dual never calls them.

Measured on supra 5070 Ti + 5060 Ti, `UD-Q4_K_XL` + `mtp.safetensors`, device-side EP route already in tree:

| | kernel | 32k unique filled tg | verify |
|---|---|---|---|
| default `g_group_attn=true` | `attn_multi` | **64** (under IQ4 59) | 42 ms |
| `IK2_GROUP=1` | flash (`T<=8` skips multi) | **81** | 32 ms |
| empty 262k + `IK2_GROUP=1` | flash | **104** | 30 ms |

IQ4_XS llama.cpp seat claims **tg 59** (DSpark n=3, 262k). Bar to beat at **filled** depth.

---

## 1. Already landed (do not redo)

Uncommitted in this clone (keep it):

| Patch | Files | Result |
|---|---|---|
| mmap dual, skip WILLNEED, fadvise DONTNEED, 256 MiB room | `ik2_main.cu` | Anon ~160 MiB; page cache 14→1.8 GiB |
| `inkling2_loop.*` keys + loop tensor name map | `ik2_main.cu` | GGUF loads |
| EP `T<=8`, `--draft` on 2 GPUs | `ik2_main.cu` | DSpark works |
| Q8 dense auto on `dual()` | `ik2_main.cu` | +0.9 GiB GPU0; **hurts tg** — use `IK2_DENSE=f16` |
| Graphs v1 (pinned staging + 3 graphs) | `ik2_main.cu` | **0 tg win** (route sync still outside) |
| Device-side route `tab_ep` + `ep_compact` | `ik2_main.cu` + `ik2_kernels.{cu,cuh}` | empty 262k 100 tg; 32k unique 64→ then 81 with GROUP=1 |
| Pipeline full overlap | refused | `W.h` written by combine; next attn reads it |

P2P is **dead** (`cudaDeviceCanAccessPeer=0`, PHB). Do not chase.

---

## 2. Product command (use this for every bench)

```
ssh tisaak@192.168.88.14
export LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64:$LD_LIBRARY_PATH
env IK2_EP=1 IK2_EP_SPLIT=0.65 IK2_DENSE=f16 IK2_DRAFT_CTX=8192 IK2_SLOT_KEEP=0 IK2_GROUP=1 \
  /home/tisaak/github/maya-inkling-mini/build/maya-ik2 \
  --model /DATA/llm/maya-inkling/Inkling-2-Mini-UD-Q4_K_XL.gguf \
  --draft /DATA/llm/maya-inkling/mtp.safetensors \
  --gpus 2 --max-context 262144 --chunk 2048 --gen 64 \
  --tokens @/tmp/maya-32k-uniq.ids
```

Prompt files on supra: `/tmp/maya-prompt.ids` (61-token code), `/tmp/maya-8k-uniq.ids`, `/tmp/maya-32k-uniq.ids` (random unique, not repeating).

**`IK2_GROUP=1` is mandatory until this surgery lands** (`g_group_exp=true`, `g_group_attn=false` → verify uses flash, not `attn_multi`). After surgery, MMA should beat flash at depth; keep the env for A/B.

Build: turbo `cmake --build ~/github/maya-inkling-mini/build -j16 --target maya-ik2` then  
`rsync -avP build/maya-ik2 tisaak@192.168.88.14:/home/tisaak/github/maya-inkling-mini/build/maya-ik2`

---

## 3. Surgery (the only remaining tg lever)

Write-set: **`src/ik2/ik2_main.cu` only** (unless `attn_block_scratch` forces a Work alloc — still that file).

### 3a. Default dual away from `attn_multi`

Around 403 / 4535–4542:

Today: `g_group_attn = true`. `IK2_GROUP` parse:

```
g_group_exp = v == 1 || v >= 3
g_group_attn = v >= 2
```

After `g_ep` is known (~4691), if `dual()` and `IK2_GROUP` unset: **`g_group_attn = false`**. Do not change `g_group_exp`. If the user set `IK2_GROUP`, honor it.

### 3b. Wire MMA inside `run_layer_attn` (the real patch)

After `prep_qkv_r_fp8` / `_f16` (~1688–1696), **replace** the `if (T == 1) attn_decode_* else if (T<=8 && g_group_attn) attn_multi_*` block (~1705–1722) with:

```
const bool mma = g_attn_mma == 1 || (g_attn_mma == 2 && !Ly.sliding);
const int ring = Ly.sliding ? Ly.kv_cap : 0;
const int ns = Ly.sliding ? std::min(g_mma_split, std::max(1, (c.window + 63) / 64)) : g_mma_split;
const bool block_ok = T >= 2 && T <= 8 && mma && c.D == 128
                      && (T * c.H / c.KV) <= 24;   // attn_block contract, ik2_kernels.cuh:429
```

**T == 1 && mma:**  
Template: `fast_token` 2508–2526.

- `k::to_f16(q_p, HD, W.x16, g_s)` (prep already wrote float `q_p`; fast_token uses `prep_*_at` which emits q16 — either is fine).
- `attn_decode_mma_fp8(W.x16, r_p, sink_p, Ly.kc8, Ly.vc8, ring, c.H, c.KV, /*pos*/ pos_p or a 1-int device pos, 0, ns, qk_scale/k_scale, v_inv, Ly.rel_proj, c.d_rel, extent, window, W.lg, W.o, gate_p, /*out16*/ W.x16, g_s, Ly.page_table)`  
  If `pos_p` is null on sliding layers, pass `qpos=pos0` like `attn_decode_fp8` does today.
- f16 twin: `attn_decode_mma_f16`.

**T=2..8 && block_ok:**  
Template: `fast_block_launches` 2868–2881.

- Need planar q: `q_ld = HD`, `r_ld = HR`, `gate_ld = HD` (run_layer_attn’s `q_p` is `[T][H][D]`, not the packed `wqkvrg` stride).
- `prep_qkv_r_fp8` already filled K/V. Then `to_f16(q_p, T*HD, W.x16)` and  
  `attn_block_fp8(W.x16, r_p, r_ld, sink_p, Ly.kc8, Ly.vc8, ring, c.H, c.KV, pos_p, T, ns, qk_scale/k_scale, v_inv, Ly.rel_proj, c.d_rel, extent, window, W.lg, W.o, gate_p, gate_ld, W.x16, g_s, Ly.page_table)`.
- `W.pos` must hold the T positions. `pos_p` is already the device pointer passed into `run_layer_attn`. Use it. If sliding passes `pos_p=nullptr`, copy `pos0..pos0+T-1` into `W.pos` first (fast_block always has `W.pos`).

**Else (T>8, or MMA off, or block_ok false):** keep the existing flash / cublas tail (~1723+). **Do not call `attn_multi_*` when `nk` is large.** If you keep `attn_multi` at all, gate it: `nk < 2048` (or similar). Depth is the case we care about.

`g_attn_mma` is already default 1 when `D==128 && H/KV<=8` (4981). Mini qualifies. Do **not** set `g_fast` / `g_fast_want` on dual.

Scratch: MMA uses `W.lg` as scratch in `fast_token`. `attn_block_scratch(H,T,n_split)` — if `W.lg_cap` is short, die or skip to flash. Check sizes before first use.

### 3c. Out of scope

- `g_fast`, CUDA graphs v2, P2P, pipeline attn(v+1)∥experts(v) (`SURGERY_PIPE_DONE.md`: illegal).
- Q8 dense as a speed play (`IK2_DENSE=f16`).
- Prefill (`T>8`) — already ~3000 tok/s flash.
- New kernels.

---

## 4. Verify (must run, must print numbers)

Unload llama-swap: `curl -sS http://192.168.88.14:8935/unload`

Same env as §2. Report a table:

| bench | expect |
|---|---|
| 61-token prompt, `--max-context 262144 --gen 64` | ≥100 tg (was 104 with GROUP=1 flash) |
| `@/tmp/maya-8k-uniq.ids`, ctx 16384 | ≥96 tg |
| `@/tmp/maya-32k-uniq.ids`, ctx 131072 | **must beat 81** (flash). Target: verify <32 ms. |
| `IK2_ATTN_MMA=0` A/B on 32k unique | should match old flash or multi — proves MMA is doing work |

Also grep the 32k log: no `die(`, 0 expert misses on decode. Generated tokens on the **code** prompt (`/tmp/maya-prompt.ids`) should still start `1160,7,37777,...` (same as pre-MMA). Random unique prompts may look like garbage; ignore.

If 32k unique **regresses** vs 81, revert the `T>1` branch to flash (`IK2_GROUP=1` path) and keep MMA for `T==1` only.

---

## 5. Compile host

turbo: CUDA 13.4, `sm_120`, no ninja (Unix Makefiles). `ccache` on. supra nvcc is 13.3 — do not compile there (16 GB RAM).

---

## 6. Decision log (do not reopen)

| | |
|---|---|
| P2P | NS / `CanAccessPeer=0` |
| Graphs wrapping EP | 86.4 vs 86.5 tok/s |
| Q8 dense | +0.9 GiB GPU0, tg 89→67 |
| Full layer pipeline | `W.h` RAW after combine |
| `IK2_EP_SPLIT` | **0.65** for tg (0.62 was VRAM-only) |
| Repeating 61-token “depth” | cheat (7 tok/step). Use unique ids. |
