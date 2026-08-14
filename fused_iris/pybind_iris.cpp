#include <torch/extension.h>

#include "include/fused_iris_kernel.cuh"

#include <tuple>
#include <vector>

#define CHECK_CUDA(x) AT_ASSERTM((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  AT_ASSERTM((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

namespace {

std::tuple<double, double, double> advance_psi(
    double psi_1_prev,
    double psi_2_prev,
    double psi_3_prev,
    double beta1,
    double beta2,
    double beta3) {
  return {
      beta1 * psi_1_prev + 1.0,
      beta2 * psi_2_prev + 1.0,
      beta3 * psi_3_prev + 1.0,
  };
}

void check_single(
    const at::Tensor& params,
    const at::Tensor& grads,
    const at::Tensor& grad_ests,
    const at::Tensor& var_ests,
    const at::Tensor& max_var_ests,
    const at::Tensor& innov_residuals,
    bool amsgrad) {
  CHECK_INPUT(params);
  CHECK_INPUT(grads);
  CHECK_INPUT(grad_ests);
  CHECK_INPUT(var_ests);
  CHECK_INPUT(innov_residuals);
  const auto n = params.numel();
  AT_ASSERTM(grads.numel() == n, "grads / params numel mismatch");
  AT_ASSERTM(grad_ests.numel() == n, "grad_ests / params numel mismatch");
  AT_ASSERTM(var_ests.numel() == n, "var_ests / params numel mismatch");
  AT_ASSERTM(innov_residuals.numel() == n, "innov_residuals / params numel mismatch");
  if (amsgrad) {
    CHECK_INPUT(max_var_ests);
    AT_ASSERTM(max_var_ests.numel() == n, "max_var_ests / params numel mismatch");
  }
}

}  // namespace

std::tuple<double, double, double> iris_fused_cuda(
    at::Tensor params,
    at::Tensor grads,
    at::Tensor grad_ests,
    at::Tensor var_ests,
    at::Tensor max_var_ests,
    at::Tensor innov_residuals,
    at::Tensor step_tensor,
    double psi_1_prev,
    double psi_2_prev,
    double psi_3_prev,
    double lr,
    double beta1,
    double beta2,
    double beta3,
    double wd,
    double eps,
    double rho,
    bool use_clipping,
    bool amsgrad) {
  check_single(
      params, grads, grad_ests, var_ests, max_var_ests, innov_residuals, amsgrad);

  auto [psi_1, psi_2, psi_3] = advance_psi(
      psi_1_prev, psi_2_prev, psi_3_prev, beta1, beta2, beta3);
  step_tensor.add_(1);

  fused_iris_cuda(
      params,
      grads,
      grad_ests,
      var_ests,
      max_var_ests,
      innov_residuals,
      static_cast<float>(lr),
      static_cast<float>(1.0 / psi_1),
      static_cast<float>(1.0 / psi_2),
      static_cast<float>(1.0 / psi_3),
      static_cast<float>(beta2),
      static_cast<float>(wd),
      static_cast<float>(eps),
      static_cast<float>(rho),
      use_clipping,
      amsgrad);

  return {psi_1, psi_2, psi_3};
}

std::tuple<double, double, double> iris_multi_tensor_fused_cuda(
    std::vector<at::Tensor> params,
    std::vector<at::Tensor> grads,
    std::vector<at::Tensor> grad_ests,
    std::vector<at::Tensor> var_ests,
    std::vector<at::Tensor> max_var_ests,
    std::vector<at::Tensor> innov_residuals,
    std::vector<at::Tensor> step_tensors,
    double psi_1_prev,
    double psi_2_prev,
    double psi_3_prev,
    double lr,
    double beta1,
    double beta2,
    double beta3,
    double wd,
    double eps,
    double rho,
    bool use_clipping,
    bool amsgrad) {
  const auto n = params.size();
  AT_ASSERTM(grads.size() == n, "grads size mismatch");
  AT_ASSERTM(grad_ests.size() == n, "grad_ests size mismatch");
  AT_ASSERTM(var_ests.size() == n, "var_ests size mismatch");
  AT_ASSERTM(innov_residuals.size() == n, "innov_residuals size mismatch");
  AT_ASSERTM(step_tensors.size() == n, "step_tensors size mismatch");
  if (amsgrad) {
    AT_ASSERTM(max_var_ests.size() == n, "max_var_ests size mismatch");
  }

  auto [psi_1, psi_2, psi_3] = advance_psi(
      psi_1_prev, psi_2_prev, psi_3_prev, beta1, beta2, beta3);
  if (n == 0) {
    return {psi_1, psi_2, psi_3};
  }

  for (size_t i = 0; i < n; ++i) {
    CHECK_INPUT(params[i]);
    CHECK_INPUT(grads[i]);
    CHECK_INPUT(grad_ests[i]);
    CHECK_INPUT(var_ests[i]);
    CHECK_INPUT(innov_residuals[i]);
    if (amsgrad) {
      CHECK_INPUT(max_var_ests[i]);
    }
    step_tensors[i].add_(1);
  }

  std::vector<std::vector<at::Tensor>> tensor_lists = {
      params, grads, grad_ests, var_ests, innov_residuals};
  if (amsgrad) {
    tensor_lists.push_back(max_var_ests);
  }

  auto noop_flag = at::zeros(
      {1}, params[0].options().dtype(at::kInt));
  constexpr int kChunkSize = 2048 * 32;

  multi_tensor_iris_cuda(
      kChunkSize,
      noop_flag,
      tensor_lists,
      static_cast<float>(lr),
      static_cast<float>(1.0 / psi_1),
      static_cast<float>(1.0 / psi_2),
      static_cast<float>(1.0 / psi_3),
      static_cast<float>(beta2),
      static_cast<float>(wd),
      static_cast<float>(eps),
      static_cast<float>(rho),
      use_clipping,
      amsgrad);

  return {psi_1, psi_2, psi_3};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("iris_fused_cuda", &iris_fused_cuda,
        "Fused IRIS update (single tensor)");
  m.def("iris_multi_tensor_fused_cuda", &iris_multi_tensor_fused_cuda,
        "Fused IRIS update (multi tensor, one launch)");
}
