/* Single-tensor fused IRIS kernel + launcher.
   Layout matches sail-sg/Adan fused_adan_kernel.cu:
   this file owns __global__ iris_kernel and fused_iris_cuda().
   Python bindings live in pybind_iris.cpp.
*/
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "include/device_utils.cuh"
#include "include/fused_iris_kernel.cuh"
#include "include/type_shim.h"

template <typename scalar_t>
__global__ void iris_kernel(
    scalar_t* __restrict__ params,
    const scalar_t* __restrict__ grads,
    scalar_t* __restrict__ grad_ests,
    scalar_t* __restrict__ var_ests,
    scalar_t* __restrict__ max_var_ests,
    scalar_t* __restrict__ innov_residuals,
    const int64_t numel,
    const scalar_t lr,
    const scalar_t psi_inv_1,
    const scalar_t psi_inv_2,
    const scalar_t psi_inv_3,
    const scalar_t beta2,
    const scalar_t wd,
    const scalar_t eps,
    const scalar_t rho,
    const bool _clip,
    const bool amsgrad
  ){
  // initialize tensor numbers ONCE 
  const scalar_t t0 = scalar_t(0);
  const scalar_t t1 = scalar_t(1);
  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  // verify and get tensors
  for (int i = thread_id; i < numel; i += blockDim.x * gridDim.x){
    const scalar_t grad = grads[i];
    scalar_t param = params[i];
    scalar_t grad_est = grad_ests[i];
    scalar_t var_est = var_ests[i];
    scalar_t innov_res = innov_residuals[i];

    // algorithm here 
    const scalar_t innov = grad - grad_est;
    grad_est += psi_inv_1 * innov;
    grad_ests[i] = grad_est;

    _lerp(psi_inv_2, innov_res, innov);
    innov_residuals[i] = innov_res;

    const scalar_t innovated_grad = grad + beta2 * innov;
    const scalar_t curr_var_est = innovated_grad * innovated_grad;
    _lerp(psi_inv_3, var_est, curr_var_est);
    var_ests[i] = var_est;

    scalar_t denom;
    if (amsgrad){
      scalar_t max_var_est = max_var_ests[i];
      max_var_est = fmax(
        static_cast<float>(max_var_est),
        static_cast<float>(var_est)
      );
      max_var_ests[i] = max_var_est;
      denom = sqrt(static_cast<float>(max_var_est));
    }
    else {
      denom = sqrt(static_cast<float>(var_est));
    }

    // apply weight decay
    if (wd != t0){
      param = param * (t1 - wd * lr);
    }

    scalar_t update, denom_term;
    scalar_t num_term = grad_est + beta2 * innov_res;
    // clipping case
    if (_clip) {
      denom_term = denom * rho + eps;
      update = clamp(
        num_term/denom_term, 
        -t1, 
        t1
      );
    }
    else {
      denom_term = denom + eps;
      update = num_term / denom_term;
    }
    param -= lr * update;
    params[i] = param;
  }
}

void fused_iris_cuda(
    at::Tensor& params,
    at::Tensor& grads,
    at::Tensor& grad_ests,
    at::Tensor& var_ests,
    at::Tensor& max_var_ests,
    at::Tensor& innov_residuals,
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
  const int64_t total_size = params.numel();
  if (total_size == 0) {
    return;
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int block_dim = 1024;
  const int grid_dim = static_cast<int>((total_size + block_dim - 1) / block_dim);

  using namespace at;
  DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT(
      params.scalar_type(), 0, "iris_kernel",
      iris_kernel<scalar_t_0><<<grid_dim, block_dim, 0, stream>>>(
          params.data_ptr<scalar_t_0>(),
          grads.data_ptr<scalar_t_0>(),
          grad_ests.data_ptr<scalar_t_0>(),
          var_ests.data_ptr<scalar_t_0>(),
          amsgrad ? max_var_ests.data_ptr<scalar_t_0>() : nullptr,
          innov_residuals.data_ptr<scalar_t_0>(),
          total_size,
          static_cast<scalar_t_0>(lr),
          static_cast<scalar_t_0>(psi_inv_1),
          static_cast<scalar_t_0>(psi_inv_2),
          static_cast<scalar_t_0>(psi_inv_3),
          static_cast<scalar_t_0>(beta2),
          static_cast<scalar_t_0>(wd),
          static_cast<scalar_t_0>(eps),
          static_cast<scalar_t_0>(rho),
          use_clipping,
          amsgrad););

  AT_CUDA_CHECK(cudaGetLastError());
}
