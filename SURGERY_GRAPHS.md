# Dual-GPU EP CUDA graphs v1

Write-set: `src/ik2/ik2_main.cu` only. Do **not** set `g_fast` / `g_fast_want` on dual. Do **not** add kernels. Do **not** reformat the file.

Target box: P2P is **dead** (`cudaDeviceCanAccessPeer=0`). `cudaMemcpyPeerAsync` already falls back through the host. Make that explicit with pinned staging, then capture two per-device graphs.

## 1. Pinned staging (always, when `dual()`)

After EP events are created (~4704), on GPU 0:

```
cudaHostAlloc ep_x_pin  (8 * C * 4)     // T<=8
cudaHostAlloc ep_y_pin  (64 * C * 4)    // S<=64 GPU1 rows
```

In the EP branch (`g_ep && T <= 8 && S <= 64`, ~3636), **replace**:

- `cudaMemcpyPeerAsync(Wd.xq, 1, dev_work(0).xq, 0, T*C*4, g_s)`
  with D2H `dev_work(0).xq → ep_x_pin` on GPU0 stream, then H2D `ep_x_pin → Wd.xq` on GPU1 stream (after `g_ep_x` wait).
- `cudaMemcpyPeerAsync(dev_work(0).y32 + S*C, 0, Wd.y32, 1, n*C*4, g_s)`
  with D2H `Wd.y32 → ep_y_pin` on GPU1, then H2D `ep_y_pin → dev_work(0).y32 + S*C` on GPU0 (before combine).

Keep `g_ep_x` / `g_ep_done` event stitching. Do not call `cudaDeviceEnablePeerAccess`.

## 2. Graphs (T=1..8, lazy)

Two `FastGraph` arrays, keyed by T: `g_ep_g0[9]`, `g_ep_g1[9]`.

Capture **after** host routing + `Ed.get()` have filled `blob_d` / `tok_d` / `sg_d` / `su_d` / `pack_h` (those H2D copies of pack tables stay **outside** the graph — they change which pointers are in the pack; wait: pack_h contents change every token but the **address** is stable. Include `cudaMemcpyAsync(pk_d, pk, 1536, H2D)` **inside** the graph if `pk`/`pk_d` are the same pointers every token for that T. They are `Wd.pack_h[ps]` ring — **ring slot changes**. Do **not** put pack H2D in the graph.

So capture only:

**g1 (GPU1 stream `g_dev[1].s`):**
- H2D `ep_x_pin → Wd.xq` (T*C)
- `experts_gate_up_swiglu` + `experts_down` for GPU1's n picks (use the same kernel calls as now)
- D2H `Wd.y32 → ep_y_pin` (use max n = S, copy S*C always so the graph shape is fixed; extra rows ignored)

**g0 (GPU0 stream `g_dev[0].s`):**
- GPU0 expert kernels (same as now)
- H2D `ep_y_pin → y32 + S*C` (S*C, fixed)
- combine `moe_combine_f32` + `conv_rec` + `conv4_res` (same as now)

**Fixed-shape rule:** the current code does `if (!n) continue` and `n` varies. For the graph, launch experts with `m = n_d[d]` as today on **first capture**, then **invalidate and recapture when n_d[0] or n_d[1] changes**. Store `int g_ep_n0[9], g_ep_n1[9]` at capture. If this token's n_d differs, destroy exec+graph for that T and recapture. (Accept rate: most decode steps share n. Fine.)

Capture with `cudaStreamBeginCapture` / `EndCapture` / `Instantiate` like `g_fgraph` (~2618–2633). Destroy on recapture like `:2325`.

**Host launch per layer after pack H2D:**

```
record g_ep_x on s0
wait s1 on g_ep_x
launch g_ep_g1[T] on s1
record g_ep_done on s1
launch g_ep_g0[T] on s0   // GPU0 experts can overlap GPU1 if they don't need y yet
wait s0 on g_ep_done      // before the H2D y / combine — those are INSIDE g0, so:
```

If combine is inside g0, g0 must wait for GPU1 y first. Then g0 cannot overlap GPU1 experts.

**Correct stitch:**

```
record g_ep_x on s0
wait s1 on g_ep_x
launch g1 on s1           // GPU1 experts + D2H y
launch g0_exp on s0       // GPU0 experts only  — overlap
record g_ep_done on s1
wait s0 on g_ep_done
launch g0_comb on s0      // H2D y + combine + conv
```

That's **three** graphs: `g0_exp[T]`, `g1[T]`, `g0_comb[T]`. Do that.

`IK2_GRAPH=0` (`g_graph` already parsed): skip capture, run the current kernel sequence with pinned staging only.

## 3. Out of scope

- `g_fast`, `fast_token`, `fast_block`, `g_fgraph`
- `group_route` / `experts_block_*` (single-GPU block path)
- New files, CMake, docs except you may append 10 lines to `SURGERY_DONE.md`
- Compile

## Done when

`rg` shows pinned staging and `cudaGraphLaunch` in the EP branch. `g_fast_want` still only set inside `n_gpu == 1`. Write `SURGERY_GRAPHS_DONE.md` with line numbers.
