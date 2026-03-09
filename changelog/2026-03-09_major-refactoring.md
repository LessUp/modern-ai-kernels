# Major Refactoring - v2.0.0

Date: 2026-03-09

## Bug Fixes

### MemoryPool lifecycle bug (Critical)
- `clear()` was erasing `allocated_sizes_` for ALL entries including in-use blocks, causing use-after-free
- `deallocate()` kept stale entries in `allocated_sizes_` instead of moving to freed tracking
- Added `freed_sizes_` map to properly track pool-returned vs in-use blocks
- `trim()` also fixed to use the correct tracking map

### atomicMin/atomicMax for negative floats (Critical)
- `compute_quant_params_kernel` in `fusion.hpp` used `atomicMin`/`atomicMax` on `float`-as-`int`, which gives incorrect results for negative values due to IEEE 754 sign-magnitude representation
- Replaced with proper CAS-based `atomicMinFloat`/`atomicMaxFloat` that handle all sign combinations correctly

## Architecture Improvements

### Warp reduction utilities extracted to `core/warp_utils.hpp`
- Moved `warp_reduce_max`, `warp_reduce_sum` from `softmax.hpp` to dedicated `core/warp_utils.hpp`
- Added `warp_reduce_min`, `warp_broadcast` utilities
- Added block-level reduction helpers: `block_reduce_sum`, `block_reduce_max`
- `normalization.hpp` no longer depends on `softmax.hpp` (was only included for warp utils)
- Backward-compatible `using` declarations in `softmax.hpp`

## Kernel Optimizations

### FlashAttention kernel rewrite
- Moved output accumulator (`o_acc[HEAD_DIM]`) from per-thread registers to shared memory — eliminates 256 bytes/thread register pressure
- Added per-row running max/sum (`row_m`, `row_l`) in shared memory
- Cooperative tile loading using linearized `tid` instead of `ty`/`tx` partitioning
- V accumulation now distributes HEAD_DIM across threads in the x dimension
- Reduced default BLOCK_M/BLOCK_N from 64 to 32 to fit shared memory budget

### Tensor::fill GPU kernel
- Replaced host-memory roundtrip for non-byte types with a proper GPU fill kernel
- Added `detail::fill_kernel` template in `tensor.hpp`

## Version
- Bumped project version from 1.0.0 to 2.0.0
