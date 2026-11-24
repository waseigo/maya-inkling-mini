# Pipeline GPU0 attn(l+1) with GPU1 experts(l)

Do **not** implement until SURGERY_ROUTE is merged and benched.

Idea: after device-side route, GPU 0's attention for visit v+1 can overlap GPU 1's expert GEMMs for visit v. Residual `W.h` / `W.xq` must not be overwritten until EP of v has consumed xq (already copied to GPU 1 / pin).

Likely: double-buffer `xq`/`h` on GPU 0. Record event after attn(v)+route(v)+xq staging; GPU1 experts(v) wait on that; GPU0 starts attn(v+1) after its own experts(v)+combine, not after GPU1.

Only legal if ROUTE patch removed the host sync. Loop visits (6–17 ×3) make v+1 sometimes a later pass of the same layer — bind_visit must be correct.

Write-set: `src/ik2/ik2_main.cu` only. Estimate ~80–150 LOC.
