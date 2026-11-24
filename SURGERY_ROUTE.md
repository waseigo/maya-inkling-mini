# Dual-GPU device-side route (drop the 48× cudaStreamSynchronize)

Write-set:
- `src/ik2/ik2_kernels.cuh` (declare)
- `src/ik2/ik2_kernels.cu` (one small kernel)
- `src/ik2/ik2_main.cu`

Do **not** set `g_fast` / `g_fast_want`. Do **not** reformat.

## Why

EP decode still `cudaMemcpyAsync` route D2H + `cudaStreamSynchronize` (`ik2_main.cu` ~3508–3516) every layer so the host can `ep_owner` + `Ed.get()`. Graphs v1 did not touch this. Measured 86 tok/s at 262k. This patch is the remaining tg lever.

## 1. `Layer::tab_ep[2]`

Next to `tab_blob` (~259): `const uint8_t** tab_ep[2] = {};`

After EP warm fill (`g_ep`, the loop that copies owned experts, ~5638+), **per GPU d**, for each layer `l` whose owned experts are all in that GPU's cache:

```
std::vector<const uint8_t*> tab(c.E, nullptr);
for e in 0..E-1:
  if ep_owner(l,e)==d && EC.has(l,e): tab[e] = slot pointer (same as resident path ~5688)
Ly.tab_ep[d] = dalloc<const uint8_t*>(c.E) on device d
cudaMemcpy tab_ep[d]
```

If any owned expert is missing, leave `tab_ep[d]` null for that layer and the EP branch keeps today's host path for that layer.

Also copy `tab_sc[0..2]` if not already on the device (resident path does `dev_f32(Ly.scale[r])`). On dual, `tab_sc` may be unset — set them once on GPU 0 and again on GPU 1 if scales live in host vectors (or duplicate).

## 2. Kernel `ep_compact`

`ik2_kernels.cuh`:

```
void ep_compact(const int* top, const float* w, const uint8_t* owner, int S, int gpu,
                int* top_out, float* w_out, int* n_out, cudaStream_t s);
```

`owner` is device `[E]` uint8 (0/1). Compact picks where `owner[top[i]]==gpu`. Write count to `n_out[0]`.

Implement in `ik2_kernels.cu`. Keep it short (one block, S<=64).

At load, per layer, upload `uint8_t owner[E]` = `ep_owner(l,e)` to `Ly.ep_own` (new `uint8_t* ep_own` on Layer, GPU 0; GPU 1 can use the same if we copy, or upload on both).

## 3. EP branch when `Ly.tab_ep[0] && Ly.tab_ep[1]`

Skip route D2H and `cudaStreamSynchronize` for that layer.

Then:
1. Pin-copy `xq` to GPU 1 (already have `g_ep_x_pin`).
2. Pin-copy `route_d` (top+w, `S*8` bytes) to GPU 1.
3. Each GPU d: `ep_compact` → `group_route` with `Ly.tab_ep[d]` → `experts_gate_up_swiglu` + `experts_down` (same as now, device blob pointers from group_route). Use `experts_block_*` **only** if `k::experts_mma_fits` (NVFP4 layers). Q8 last-4 layers stay `experts_gate_up_swiglu`.
4. y back through `g_ep_y_pin`, combine + conv on GPU 0 (same as graphs v1 stitch).

If `tab_ep` is null, keep the current host `Ed.get` path.

`g_blk` is single-GPU today (`ensure_block_bufs`). Allocate per-GPU compact/group buffers (`blob/n/tok/row/sg/su/wt`, S=64) on each device in dual init. Do not call `ensure_block_bufs` (it host-allocs huge logits tables).

## 4. Out of scope

- Pipeline attention(l+1) with experts(l) — later file
- `g_fast`, `fast_token`
- Compile

## Done

`SURGERY_ROUTE_DONE.md` with line numbers. `rg cudaStreamSynchronize` in the EP-with-tab_ep path should be gone.
