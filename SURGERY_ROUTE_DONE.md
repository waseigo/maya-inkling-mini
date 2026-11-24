# SURGERY_ROUTE_DONE — dual-GPU EP: the route and the experts on the devices

Repo: maya-inkling-mini. Write-set as asked: `src/ik2/ik2_kernels.cuh` (+11), `src/ik2/ik2_kernels.cu` (+41),
`src/ik2/ik2_main.cu` (+170). Not compiled, not run, not committed. No `g_fast` / `g_fast_want` / `fast_token`.

## Changed lines (final files)

| # | Change | File:lines |
|---|--------|-----------|
| 1 | `ep_compact` / `ep_slots` declared | `ik2_kernels.cuh:471–478` |
| 2 | `group_route_kernel`: a slot whose expert is `< 0` (ep_compact's mask) makes no group | `ik2_kernels.cu:5134` |
| 3 | `ep_compact_kernel` / `ep_compact`: a GPU's share of a window's picks, count in `n_out[0]` | `ik2_kernels.cu:6539–6562` |
| 4 | `ep_slots_kernel` / `ep_slots`: the combine's (row, weight) table, built on the device | `ik2_kernels.cu:6563–6578` |
| 5 | `Layer::tab_ep[2]`, `Layer::ep_own[2]`, `Layer::sc_ep[2][3]` (next to `tab_blob` / `tab_sc`) | `ik2_main.cu:262–267` |
| 6 | `g_ep_tab` (the device route is on) and `g_ep_tab_miss[2]` (the miss counts the tables were built at) | `ik2_main.cu:383–384` |
| 7 | `EpDev` + `g_ep_dev[2]`: a GPU's masked picks, group_route's groups, and GPU 0's combine table | `ik2_main.cu:2278–2294` |
| 8 | `build_ep_tab(M, d)`: each layer's owned experts' slots → `tab_ep[d]`, false when one is not resident | `ik2_main.cu:3435–3465` |
| 9 | `forward`: rebuild the tables when the caches' miss counts moved (a verify window's insert) | `ik2_main.cu:3476–3487` |
| 10 | `ep_dev` predicate, and the route D2H + `cudaStreamSynchronize` + `pop_count` + dump / trace now behind `!ep_dev` | `ik2_main.cu:3598–3625` |
| 11 | The new branch: x and route to GPU 1 over pinned memory (one event), per GPU `ep_compact` → `group_route` → experts, stitch + combine + convs on GPU 0 | `ik2_main.cu:3732–3790` |
| 12 | Dual init: `EpDev`'s buffers, 64 slots a GPU (`blob/n/tok/row/sg/su/wt/inv`, and `id/wtc` on GPU 0) | `ik2_main.cu:4930–4940` |
| 13 | Load, per GPU: owner table `[E]`, the scales' copies, then `build_ep_tab`; the verdict + one line of print | `ik2_main.cu:5817–5830`, `5917–5924` |

## What a layer does now (EP, T ≤ 8, S = T·K ≤ 64, tables present)

1. `route()` as before, on GPU 0. Then `xq` and `route_d` (`S*8` bytes: top, then w) go to GPU 1 over the pinned
   buffers the graph path already had (`g_ep_x_pin`, `Work::route_h`), ordered by one event — the host reads neither.
2. Each GPU: `ep_compact` (its own picks; the other GPU's slots become `-1`) → `group_route` with `Layer::tab_ep[d]`
   and `Layer::sc_ep[d]` → `experts_block_gu` / `experts_block_down` when `experts_mma_fits` (the NVFP4 layers), else
   `experts_gate_up_swiglu_g` / `experts_down_g` (the Q8 layers). A member's token and row are its own slot's.
3. GPU 1's rows come back over `g_ep_y_pin` (`g_ep_done`); GPU 0 waits on it, H2Ds them into the back of `g_ep_y`,
   `ep_slots` writes the combine's table there, then `moe_combine_f32` + `conv_rec` + `conv4_res` as before.

No `cudaStreamSynchronize` in `3732–3790` (`3613` is the host path's, behind `!ep_dev`). The old branch (`3791+`, with
its `Ed.get` loop, pack tables and three graphs) is what runs when the tables are not there.

## Where this departs from the contract, and why

1. **`ep_compact` masks, it does not pack.** `group_route` reads a member's token (`i / K`) and its row (`i`) from the
   slot index, so packing a GPU's picks to the front would give every member another pick's token and row — its
   interface has no per-slot token array. So `ep_compact` keeps the window's slots, writes `-1` for the other GPU's,
   and still writes the count to `n_out[0]`.
2. **`group_route_kernel` gained one line** (`if (e < 0) return;`). Inert for every other caller (their `top[]` is all
   real experts, and `tab[e]` was read before too). Without it a masked slot would write `tok` / `row` past the last
   group member and make a group of a null blob.
3. **Launch shapes are the slot count, not the pick counts.** The pick counts are the device's now, so a graph's baked
   `n` is gone: this path launches directly (no capture, no variant lookup), and every group past the last returns at
   once (`n[g] = 0`). That is ~7–8 launches a GPU a layer (two staged copies, `ep_compact`, `group_route`, the limbs,
   gate|up, down — plus GPU 0's table, combine and convs) against three graphs before; whether that costs more than the
   stall it removes is the measurement to make, not something this file can claim.
4. **The scales are per GPU.** The contract's `tab_sc` is one device's pointer and a GPU cannot read the other's memory
   (peer access is 0 on the target box), so `Layer::sc_ep[2][3]` holds one copy a GPU (only the non-empty ones: NVFP4
   has scales, the Q8 layers do not).
5. **`ep_slots` is a second small kernel** (not in the contract). The combine's table was built by the host from the
   picks it had copied back; with the picks on the device, slot i's row is its owner's block at the slot itself and its
   weight is `w[i] * down's scale`. One block, S threads, one launch a layer.
6. **`Layer::tab_ep` is `[E]`, not `[E + 64]`**: the mask is `-1` and `group_route` never indexes a table with it.
7. **NVFP4 experts now go through the blocks' tensor-core path** (`experts_block_*`), as the contract asks. That is not
   bit-identical to the per-slot row-dot path the old branch used; expect ~1e-3 relative differences in the `y` rows
   and an unchanged argmax on all but the near-tiest tokens (the same trade the `g_blk` prompt path already makes).

## A prefill turns the path off, on purpose

The warm fill puts the experts a GPU owns into its cache (`5766+`) and the tables are built from that. Decode then
never calls `Ed.get`, so nothing evicts them. A prefill (`T > 8`) does: it takes the host path and streams experts
through the same caches, so the slots the tables point at move. `forward` compares the caches' miss counts with the
ones the tables were built at (`3476`) — on a mismatch it rebuilds, and if a layer's owned experts are no longer all
resident, `g_ep_tab` goes false and decode takes the old host path; `5921` prints which of the two is in play. Reading
moved slots would be silent wrong experts, which is worse than the old speed. Making the prefill EP-aware is the next
file.

## Left alone

- `g_fast` / `g_fast_want` / `fast_token` / `block_*` / `q_graph`, the old EP branch (`3791+`) with its graphs and pack
  tables, the TP branch, `ensure_block_bufs`, the CPU lane.
- `group_route`'s grouping itself and its other callers; `experts_block_*`; the resident path's `tab_blob` / `tab_sc`.
- The uncommitted WIP that was in the tree on entry (the graphs v1 patch, the `S <= 64` EP widening, the mmap /
  `MADV_DONTNEED` surgery) — untouched, no reformat.

## How to verify (needs a build on the box)

```bash
grep -n cudaStreamSynchronize src/ik2/ik2_main.cu | awk -F: '$1>3730 && $1<3791'   # nothing
grep -n "ep_compact\|ep_slots\|tab_ep\|g_ep_tab" src/ik2/ik2_main.cu src/ik2/ik2_kernels.*
IK2_EP=1 ./maya-ik2 ...                    # starts with "expert parallel: every owned expert resident ..."
IK2_EP=1 IK2_TRACE=/dev/null ./maya-ik2 ... # the old host path (the trace wants the picks on the host): the A/B
```

Compare the two runs' tok/s and their sampled output on the same prompt; a dump (`--dump out`) is unchanged from
before the patch only when the run is not on the device path (a dump turns it off).

## Summary (5 lines)

1. A layer whose owned experts are all resident on both GPUs now keeps its routing on the devices: no route copy, no
   `cudaStreamSynchronize`, no `Ed.get` loop and no pack table.
2. `ep_compact` splits a window's picks by owner (the other GPU's slots are `-1`), `group_route` skips them, the grouped
   expert kernels run with the slot count as the bound, and `ep_slots` builds the combine's table on GPU 0.
3. Cross-device staging stays what it was — pinned host memory ordered by events — so the host never waits on a pick.
4. A prefill that evicts the owned set turns the device route off (miss-count check) and decode falls back to the old
   host path rather than read moved slots.
5. Nothing else changed: no `g_fast`, no compile, no commit; the dirty WIP in the tree is untouched.
