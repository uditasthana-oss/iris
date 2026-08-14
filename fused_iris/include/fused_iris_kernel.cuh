#pragma once

#include <ATen/ATen.h>
#include <torch/extension.h>
#include <vector>

// CUDA launchers. DECLARE here, DEFINE in the matching .cu.
// pybind_iris.cpp is the only caller. Python never includes this file.

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
    bool amsgrad);

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
    bool amsgrad);
