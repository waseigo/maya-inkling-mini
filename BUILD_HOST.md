# BUILD_HOST — maya-ik2 compile-site inventory (sm_120 / RTX 50)

Inventory only. No compile, no `cmake`, no src edits, no weight download.
Checked 2026-11-23.

## 1. turbo (local)

| Check | Value |
|---|---|
| nvcc path | `/usr/local/cuda/bin/nvcc` |
| nvcc version | CUDA 13.4 (V13.4.92) |
| cmake | 3.31.6 (`/usr/bin/cmake`) |
| ninja | **not installed** (`command not found`) — use `Unix Makefiles` or install ninja |
| g++ | 14.2.0 (`/usr/bin/g++`) |
| CUDA toolkit dir | `/usr/local/cuda` → `/usr/local/cuda-13.4` |
| Free RAM | 62 Gi total, **18 Gi available** (8.3 Gi free + 15 Gi cache); ~64 GB kit |
| Free disk (/home) | 49 G avail of 477 G (**90% used**) |
| `-DCMAKE_CUDA_ARCHITECTURES=120` legal? | **YES** — `sm_120` present in `nvcc --help` allowed values (built on sm_100–sm_121) |

## 2. supra (`ssh -o BatchMode=yes tisaak@192.168.88.14`)

| Check | Value |
|---|---|
| nvcc path | `/usr/local/cuda-13.3/bin/nvcc` (NOT on PATH) |
| nvcc version | CUDA 13.3 (V13.3.73) |
| cmake | 3.31.6 (`/usr/bin/cmake`) |
| ninja | 1.12.1 (`/usr/bin/ninja`) |
| g++ | — (not checked; cmake 3.31.6 present) |
| CUDA toolkit dir | `/usr/local/cuda-13.3` (also `/usr/local/cuda`, `/usr/local/cuda-13` symlinks) |
| Free RAM | 15 Gi total, **248 Mi free** (13 Gi available incl. cache) |
| Free disk (/DATA/llm) | 158 G avail of 886 G (**83% used**) |
| `-DCMAKE_CUDA_ARCHITECTURES=120` legal? | **YES** — `sm_120` present in `nvcc --help` |

llama-swap `/running` not reachable on supra :8080–:8082 (empty response); no GPU job started.

## 3. Recommendation

Compile on **turbo**, then rsync the binary to supra — turbo has ~18 Gi free RAM and the newer
CUDA 13.4 toolkit versus supra's ~250 Mi free of a 15 Gi stick, and CUDA model compiles are RAM-heavy
so supra would OOM during link; the binary still runs on supra because both ship `libcudart.so.13`
(13.4→13.3 runtime is major-version compatible).
