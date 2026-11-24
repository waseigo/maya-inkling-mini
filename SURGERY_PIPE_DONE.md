# SURGERY_PIPE_DONE — the part of the visit pipeline that is legal, and the overlap that is not

Repo: maya-inkling-mini (working tree, `SURGERY_ROUTE` patch in place). Write-set as asked:
`src/ik2/ik2_main.cu` only (+9 net lines, 7800). Not compiled, not run, not committed. No `g_fast` /
`g_fast_want` / `fast_token` touched.

**Short version: the plan's headline overlap — GPU 0's attention of visit v+1 beside GPU 1's experts of
visit v — cannot be built on this graph. It is not a staging problem that double buffering fixes; the
attention of v+1 needs the visit v MoE output, and the visit v MoE output needs GPU 1's rows. The proof is
in §2. What is left of the idea — GPU 1's input off GPU 0's router, and the combine's table off the
barrier — is implemented in §1.**

## 1. Changed lines (final file)

| # | Change | Lines |
|---|--------|-------|
| 1 | `g_ep_xq`: the device route's input event, beside `g_ep_x`/`g_ep_cp`/`g_ep_done` | 385–386 |
| 2 | `ep_dev` (and `S`) decided **before** the MoE's norm, not after the route | 3575–3581 |
| 3 | The layer's input staged to the pinned buffer at the norm, one event: `g_ep_x_pin` D2H + `g_ep_xq` | 3600–3602 |
| 4 | The ep branch stages only the picks now; `g_ep_x` (route copy done) stays as the second order point | 3747–3750 |
| 5 | GPU 1: wait `g_ep_xq` → H2D of `xq`, wait `g_ep_x` → H2D of the picks | 3757–3761 |
| 6 | `ep_slots` (the combine's table) built on GPU 0 before the barrier, beside both GPUs' experts | 3766–3767 |
| 7 | `ep_slots` removed from the post-barrier tail (it delayed the combine) | 3786–3794 |
| 8 | Dual init: `g_ep_xq` created on GPU 0 (waited on by GPU 1's stream, as `g_ep_x` already was) | 4919 |

What each buys, in the barrier's terms:

1. GPU 1's copy of the layer's input is `T·C·4` bytes (up to 8 rows) and used to sit *behind* GPU 0's
   router on GPU 1's critical path (`k::route` + `gemm16` at 3605–3608). It is now queued at the norm
   (3600), so it crosses PCIe while GPU 0 routes, and GPU 1's first kernel waits only on the picks.
2. `ep_slots` needs nothing but the route (`rtop`, `rw`, `Ly.ep_own[0]`, `Ly.sc_ep[0][2]`), so it ran after
   GPU 1's rows arrived and delayed the combine. Before the barrier it runs inside the time GPU 0 spends
   waiting anyway (GPU 0's own share is the smaller one: it also runs attention, routing and the convs).

`bind_visit` (309–327) stays correct by construction: nothing added reads or writes a visit's state. The
staging reads `Work::xq` on GPU 0 and writes GPU 1's `Work::xq`; `per-layer` tables (`Ly.ep_own`,
`Ly.tab_ep`, `Ly.sc_ep`) are the layer's, not the visit's; `Ly.h_mlp` / `Ly.sc_mlp` are still used only
inside the same iteration (3793–3794). Visits 6–17 ×3 therefore behave as before: same tables, same
buffers, one visit at a time.

## 2. Refused: attention(v+1) beside experts(v). It would race and it would compute the wrong model

The visit body is one line, not two lanes. With the device route in play:

| step | where | reads | writes |
|---|---|---|---|
| attention(v) | 3572 → `run_layer_attn` 1518–1522 ("`h_in` … acts as the residual input", 1519) | `W.h` | `W.r` (= `h_in` + attn) |
| MoE norm(v) | 3583–3599 | blocks + `W.r` | `W.xq`, `W.x16` |
| route(v) | 3605–3608 | `W.x16` | `W.route_d` |
| experts(v) GPU 1 | 3756–3780 | GPU 1's `xq`, picks, tables | GPU 1's `y32` |
| rows(v) back | 3782 → event `g_ep_done` (3783) | GPU 1's `y32` | `g_ep_y_pin` |
| barrier | 3789 | — | — |
| H2D + combine(v) | 3789–3792 | `g_ep_y`, id/wt | `W.tmp` |
| conv(v) | 3793–3794 | `W.tmp`, `h_mlp`, `W.r` | **`W.h`** |
| attention(v+1) | next iteration, 3572 | **`W.h`** | `W.r`, `W.xq`, `W.q`, … |

So `attention(v+1) ⇒ combine(v) ⇒ GPU 1's rows(v)`. Issuing the attention early does not merely race;
it reads the pre-MoE stream. Concretely, if the run-ahead were added:

- **Wrong math, silently.** `W.h` at 3572 would still be visit v's stream *before* its MoE residual. No
  event or buffer can repair that: the next visit's first kernel is an RMS-norm of the stream (`rms_f16`
  1571 / `attn_res_rms_f16` 1560, 1566), so the missing half of the sum cannot be added later — the sum is
  inside a nonlinearity.
- **`W.r` clobbered under `conv4_res`.** 3794 passes `W0.r` as the residual while 3572 writes `W.r` as its
  output. Same stream on GPU 0, so the order is defined — which is exactly why the result would be
  visit v+1's attention output added to visit v's stream.
- **GPU 1's inputs rewritten mid-flight.** `EpDev` (`top`, `w`, `n`, `blob`, `gn`, `tok`, `row`, `sg`,
  `su`, `wt`, `inv`) is one set a GPU (struct 2282–2295, `g_ep_dev[2]` at 2295, its buffers 4941–4950).
  Visit v+1's `ep_compact`/`group_route` (3763–3765) would overwrite the tables that visit v's
  `experts_block_gu/_down` (3769–3780) read on the same device, and so would GPU 1's `xq`, `route_d`,
  `h16`, `gu32` (the f16 limbs) and `y32`.
- The plan's "double-buffer `xq`/`h`" covers one of those six buffers. A correct run-ahead would need a
  second set of `EpDev`, of the limbs and of `y32` on **both** GPUs, a second `g_ep_x_pin`/`g_ep_y_pin`
  pair, and it would still trip over the first bullet.

An overlay is only possible where the two GPUs' work is genuinely independent — and inside one visit only
the experts are (they already run side by side on `g_dev[0].s` and `g_dev[1].s`, 3752–3785). GPU 0's stall
at the barrier is the price of the split, not of the staging.

## 3. What is left of the intent, and what it is worth

These are the legal levers on the same critical path. None is in this write-set; listed with the files
they need so the next surgery can pick one deliberately.

1. **GPU 1's rows as a partial combine** — the return trip is `S·C·4` bytes each way (3782, 3790–3791) although
   GPU 0 needs only GPU 1's `n1` rows. A `[T][C]` partial sum would make it `T·C·4`: `K` times
   less (8 here) at any `T`, because `S = T·K`. Needs a kernel (`ep_partial`, or a masked `ep_slots`: the current one
   indexes `owner[top[i]]` and `sd[top[i]]` without a guard, so it cannot take `ep_compact`'s masked
   `top`), i.e. `src/ik2/ik2_kernels.*`. It also reassociates the K-way sum (FP32), so a dump comparison
   would show last-bit differences.
2. **Graphs for this path** — the ep route issues ~22 CUDA calls a visit (the whole point of the device
   route was to remove the stall, not the launches). Every launch here has a *fixed* shape (`S` slots, not
   the device-side counts), so three graphs a visit capture it, exactly like `g_ep_graphs` (2273) does
   for the host path. `src/ik2/ik2_main.cu` only. That was `SURGERY_GRAPHS`' question, not this file's.
3. **The split itself** (`IK2_EP_SPLIT`) — with GPU 0 also running attention, routing and the convs, the
   barrier waits on GPU 1 exactly when GPU 1's half is longer. That is a number to sweep on the box, not a
   patch.
4. **The only true attention/experts overlap is across requests** — slot A's attention beside slot B's
   experts (`g_parallel`, 282). Per-visit it does not exist.

## 4. How to verify (needs a build on the box)

```bash
grep -n "g_ep_xq" src/ik2/ik2_main.cu                       # 386, 3602, 3758, 4919: stage, record, wait, create
grep -n "ep_slots" src/ik2/ik2_main.cu                      # 3767 only in the ep branch (3786–3794 has none)
grep -n cudaStreamSynchronize src/ik2/ik2_main.cu | awk -F: '$1>3745 && $1<3796'   # nothing: the branch still never waits for the host
IK2_EP=1 ./maya-ik2 ...                                     # starts with "expert parallel: every owned expert resident"
IK2_EP=1 IK2_TRACE=/dev/null ./maya-ik2 ...                 # the host path: trace/dump turn the device route off, the A/B
```

Compare tok/s and the sampled text on the two `IK2_EP` lines (the staging change must be output-neutral: it
moves copies, it does not touch a value), and at 262k short-prompt the `IK2_EP=1` rate against the 100 tok/s
of `SURGERY_ROUTE`. The two hunk groups in `git diff` should be the only changes in this file.

## 5. Summary (5 lines)

1. GPU 1's copy of a visit's input now starts at the MoE norm, before the route, ordered by its own event
   (`g_ep_xq`); GPU 1 waits on it for the H2D and on the old `g_ep_x` for the picks.
2. The combine's table (`ep_slots`) is built before the barrier, so it no longer delays the combine behind
   GPU 1's rows.
3. The plan's overlap — attention(v+1) beside experts(v) — is refused: attention(v+1) reads `W.h`, which
   `conv4_res` writes from the combine of GPU 1's rows, and the next visit's norm is nonlinear, so the rows
   cannot be applied later either.
4. A run-ahead would also need second sets of `EpDev`, the limbs and `y32` on both GPUs, not only of
   `xq`/`h`; the two GPUs' *experts* are already concurrent, and that is the whole debt the split creates.
5. Nothing else changed: no `g_fast`, no compile, no commit, no other file, `bind_visit` untouched.
