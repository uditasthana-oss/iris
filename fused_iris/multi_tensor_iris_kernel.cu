/* Multi-tensor fused IRIS.
   Layout matches sail-sg/Adan multi_tensor_adan_kernel.cu:
   one Apex multi_tensor_apply launch for the whole model, not one
   launch per tensor. The update uses the same _lerp / clamp helpers
   as iris_kernel.
*/
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>

#include "include/device_utils.cuh"
#include "include/fused_iris_kernel.cuh"
#include "include/multi_tensor_apply.cuh"
#include "include/type_shim.h"

#define BLOCK_SIZE 512
#define ILP 4

template <typename T, bool kAmsgrad>
struct IrisFunctor {
  __device__ __forceinline__ void operator()(
      int chunk_size,
      volatile int* /*noop_gmem*/,
      TensorListMetadata<kAmsgrad ? 6 : 5>& tl,
      const float lr,
      const float psi_inv_1,
      const float psi_inv_2,
      const float psi_inv_3,
      const float beta2,
      const float wd,
      const float eps,
      const float rho,
      const bool use_clipping) {
    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    T* p = reinterpret_cast<T*>(tl.addresses[0][tensor_loc]) + chunk_idx * chunk_size;
    T* g = reinterpret_cast<T*>(tl.addresses[1][tensor_loc]) + chunk_idx * chunk_size;
    T* grad_ests = reinterpret_cast<T*>(tl.addresses[2][tensor_loc]) + chunk_idx * chunk_size;
    T* var_ests = reinterpret_cast<T*>(tl.addresses[3][tensor_loc]) + chunk_idx * chunk_size;
    T* innov_residuals = reinterpret_cast<T*>(tl.addresses[4][tensor_loc]) + chunk_idx * chunk_size;
    T* max_var_ests = nullptr;
    if (kAmsgrad) {
      max_var_ests = reinterpret_cast<T*>(tl.addresses[5][tensor_loc]) + chunk_idx * chunk_size;
    }

    n -= chunk_idx * chunk_size;

    const T t0 = T(0);
    const T t1 = T(1);
    const T lr_t = static_cast<T>(lr);
    const T psi1 = static_cast<T>(psi_inv_1);
    const T psi2 = static_cast<T>(psi_inv_2);
    const T psi3 = static_cast<T>(psi_inv_3);
    const T beta2_t = static_cast<T>(beta2);
    const T wd_t = static_cast<T>(wd);
    const T eps_t = static_cast<T>(eps);
    const T rho_t = static_cast<T>(rho);

    for (int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * ILP) {
      T r_p[ILP];
      T r_g[ILP];
      T r_m[ILP];
      T r_v[ILP];
      T r_s[ILP];
      T r_max[ILP];
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        int i = i_start + threadIdx.x + ii * blockDim.x;
        if (i < n && i < chunk_size) {
          r_p[ii] = p[i];
          r_g[ii] = g[i];
          r_m[ii] = grad_ests[i];
          r_v[ii] = var_ests[i];
          r_s[ii] = innov_residuals[i];
          if (kAmsgrad) {
            r_max[ii] = max_var_ests[i];
          }
        } else {
          r_p[ii] = t0;
          r_g[ii] = t0;
          r_m[ii] = t0;
          r_v[ii] = t0;
          r_s[ii] = t0;
          r_max[ii] = t0;
        }
      }
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        const T innov = r_g[ii] - r_m[ii];
        r_m[ii] += psi1 * innov;
        _lerp(psi2, r_s[ii], innov);

        const T innovated_grad = r_g[ii] + beta2_t * innov;
        _lerp(psi3, r_v[ii], innovated_grad * innovated_grad);

        T denom;
        if (kAmsgrad) {
          r_max[ii] = static_cast<T>(
              fmax(static_cast<float>(r_max[ii]), static_cast<float>(r_v[ii])));
          denom = static_cast<T>(sqrt(static_cast<float>(r_max[ii])));
        } else {
          denom = static_cast<T>(sqrt(static_cast<float>(r_v[ii])));
        }

        if (wd_t != t0) {
          r_p[ii] = r_p[ii] * (t1 - wd_t * lr_t);
        }

        const T num_term = r_m[ii] + beta2_t * r_s[ii];
        T update;
        if (use_clipping) {
          update = clamp(num_term / (denom * rho_t + eps_t), -t1, t1);
        } else {
          update = num_term / (denom + eps_t);
        }
        r_p[ii] -= lr_t * update;
      }
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        int i = i_start + threadIdx.x + ii * blockDim.x;
        if (i < n && i < chunk_size) {
          p[i] = r_p[ii];
          grad_ests[i] = r_m[ii];
          var_ests[i] = r_v[ii];
          innov_residuals[i] = r_s[ii];
          if (kAmsgrad) {
            max_var_ests[i] = r_max[ii];
          }
        }
      }
    }
  }
};

void multi_tensor_iris_cuda(
    int chunk_size,
    at::Tensor noop_flag,
    std::vector<std::vector<at::Tensor>> tensor_lists,
    float lr,
    float psi_inv_1,
    float psi_inv_2,
    float psi_inv_3,
    float beta2,
    float wd,
    float eps,
    float rho,
    bool use_clipping,
    bool amsgrad) {
  TORCH_CHECK(!tensor_lists.empty(), "tensor list cannot be empty");
  if (tensor_lists[0].empty()) {
    return;
  }

  using namespace at;
  if (amsgrad) {
    TORCH_CHECK(tensor_lists.size() == 6, "amsgrad expects 6 tensor lists");
    DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT(
        tensor_lists[0][0].scalar_type(), 0, "iris_multi_tensor",
        multi_tensor_apply<6>(
            BLOCK_SIZE,
            chunk_size,
            noop_flag,
            tensor_lists,
            IrisFunctor<scalar_t_0, true>(),
            lr,
            psi_inv_1,
            psi_inv_2,
            psi_inv_3,
            beta2,
            wd,
            eps,
            rho,
            use_clipping););
  } else {
    TORCH_CHECK(tensor_lists.size() == 5, "expected 5 tensor lists");
    DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT(
        tensor_lists[0][0].scalar_type(), 0, "iris_multi_tensor",
        multi_tensor_apply<5>(
            BLOCK_SIZE,
            chunk_size,
            noop_flag,
            tensor_lists,
            IrisFunctor<scalar_t_0, false>(),
            lr,
            psi_inv_1,
            psi_inv_2,
            psi_inv_3,
            beta2,
            wd,
            eps,
            rho,
            use_clipping););
  }

  AT_CUDA_CHECK(cudaGetLastError());
}
