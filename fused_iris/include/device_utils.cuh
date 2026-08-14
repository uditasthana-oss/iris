#pragma once
#include <cuda_runtime.h>

// Tiny device helpers. These are templates, so the bodies MUST live in a
// header — the compiler needs to see them wherever iris_kernel calls them.

template <typename scalar_t>
__device__ __forceinline__ scalar_t clamp(
    scalar_t val,
    scalar_t lo,
    scalar_t hi
) {
    return static_cast<scalar_t>(
        fminf(static_cast<float>(hi),
              fmaxf(
                static_cast<float>(lo), 
                static_cast<float>(val)
            )
        )
    );
}

template <typename scalar_t>
__device__ __forceinline__ void _lerp(
    scalar_t alpha,
    scalar_t& state,
    scalar_t value
) {
    state += alpha * (value - state);
}
