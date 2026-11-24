# SURGERY_GRAPHS_DONE — dual-GPU EP: pinned staging + three graphs a visit

Repo: maya-inkling-mini, write-set `src/ik2/ik2_main.cu` only. Not compiled, not run, not committed.

## Changed lines (final file, 7621 lines)

| # | Change | Lines |
|---|--------|-------|
| 1 | `EpGraphs` (n0/n1 variants + `g1`/`g0x`/`g0c`), `g_ep_graphs[slot][visit][T]`, pinned `g_ep_x_pin`/`g_ep_y_pin`, pinned+device pack tables `g_ep_pk_h[2]`/`g_ep_pk_d[2]`, `g_ep_y` | 2251–2268 |
| 2 | Dual init: `cudaHostAlloc` 8·C·4 + 64·C·4, the two 2 KB pack tables a GPU, `g_ep_y` (2·64·C floats on GPU 0), `g_ep_graphs.resize` | 4784–4794 |
| 3 | EP branch rewritten (was `cudaMemcpyPeerAsync` + the pack ring) | 3655–3794 |
| 4 | x → GPU 1: D2H `xq → g_ep_x_pin` on s0, `g_ep_x` recorded (queued before the routing) | 3666–3667 |
| 5 | Picks, packs, one H2D a GPU of 2048 B; the combine's id/wt ride at +768/+1536 | 3668–3712 |
| 6 | Graph variant lookup and recapture (destroy after both streams sync) | 3713–3736 |
| 7 | Kernels: `expert_rows` (one launch a GPU, m = n_d[d]), `ep_g1`, `ep_g0x`, `ep_g0c` | 3738–3761 |
| 8 | `replay`: `cudaStreamBeginCapture` / `EndCapture` / `Instantiate` / `cudaGraphLaunch`, or the body inline when `IK2_GRAPH=0` or `g_rec` | 3762–3778 |
| 9 | The stitch: `g_ep_x` → GPU 1 (with `g_ep_cp[1]`), `g_ep_done`, GPU 0's experts in between, `g_ep_cp[0]` + y H2D + combine after | 3779–3791 |

## Where this departs from the contract, and why

1. **Keyed by slot+visit+T+n0+n1, eight variants a visit**, not `g_ep_n0[9]`/`g_ep_n1[9]`. A graph bakes the kernel arguments, and `Ly.eb` / `Ly.ty` / `h_mlp` / `sc_mlp` / the conv records are a *visit's*; keyed by T alone the first layer's pointers would replay for all 43. And per visit the pick counts change most tokens, so one variant a visit would recapture 48×3 graphs a token.
2. **Pack H2D stays outside the graph** (as asked), into a table at a fixed address: the contract's own reason (the ring slot moves) makes the graph's baked `pk_d` stale otherwise. One pinned+device table a GPU, uploaded whole each visit. The combine's id/wt go in the same table's unused +768/+1536, so there is no second upload or buffer.
3. **`g_ep_y` instead of `W.y32`**: `W.y32` is 64·C floats, and GPU 1's rows start at S·C, so S = 64 needs 128·C. Dedicated 2·64·C buffer on GPU 0.
4. **One expert launch a GPU, m = n_d[d]** (hits then misses in the pack, one `g_ep_cp` wait before them) rather than the old hits/misses two-part split: a graph needs a fixed shape. `IK2_GRAPH=0` takes the same path, so it no longer starts the hits before the copies land.
5. **No capture while `g_rec`** (a verify run's conv records are nodes, so graphs must not outlive a mode change).

## Left alone (out of scope, or pre-existing)

- `cudaDeviceEnablePeerAccess` at 4769–4771: pre-existing, guarded by `cudaDeviceCanAccessPeer` (0 on the target box), TP-only. Nothing new calls it; the EP branch uses no peer copy.
- TP's `cudaMemcpyPeerAsync` at 3596/3625, `g_fast` / `g_fast_want` / `fast_token` / `block_*`, `group_route`.
- The WIP's `S <= 64` EP widening and its `tok` rows (tree was dirty on entry; untouched).

## How to verify (needs a build on the box)

```bash
rg -n "cudaGraphLaunch|cudaHostAlloc|g_ep_x_pin" src/ik2/ik2_main.cu          # EP branch now stages + graphs
rg -n "cudaMemcpyPeerAsync" src/ik2/ik2_main.cu                                # should list only 2058/2064/2152/3596/3625
IK2_EP=1 IK2_GRAPH=0 ./maya-ik2 ...                                            # the same numbers as before the change (staging only)
IK2_EP=1 ./maya-ik2 ...                                                        # graphs; compare tok/s and a dump against the line above
```

Watch for: `cudaErrorStreamCaptureEmpty` (a variant captured with n_d = 0 — guarded), and eviction cost (two stream syncs) if decode keeps inventing new (n0, n1).
