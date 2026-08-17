"""Fused CUDA kernel wrapper for IRIS optimizer."""

import torch
from typing import List, Optional, Tuple

_fused_kernel = None
_fused_available = False

try:
    from .fused_iris import fused_iris_kernel
    _fused_kernel = fused_iris_kernel
    _fused_available = fused_iris_kernel is not None
except ImportError:
    try:
        from . import fused_iris_kernel
        _fused_kernel = fused_iris_kernel
        _fused_available = fused_iris_kernel is not None
    except ImportError:
        _fused_available = False


def is_fused_available() -> bool:
    """Check if fused CUDA kernel is available."""
    return _fused_available


def _collect_fused_tensors(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    grad_estimates: List[torch.Tensor],
    variance_estimates: List[torch.Tensor],
    max_variance_estimates: List[torch.Tensor],
    innovation_residuals: List[torch.Tensor],
    state_steps: List[torch.Tensor],
    amsgrad: bool,
) -> Tuple[
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
]:
    params_with_grad = []
    grads_with_grad = []
    grad_ests_with_grad = []
    var_ests_with_grad = []
    max_var_ests_with_grad = []
    innov_res_with_grad = []
    steps_with_grad = []

    for i, grad in enumerate(grads):
        if grad is None:
            continue
        params_with_grad.append(params[i])
        grads_with_grad.append(grad)
        grad_ests_with_grad.append(grad_estimates[i])
        var_ests_with_grad.append(variance_estimates[i])
        if amsgrad:
            max_var_ests_with_grad.append(max_variance_estimates[i])
        innov_res_with_grad.append(innovation_residuals[i])
        steps_with_grad.append(state_steps[i])

    return (
        params_with_grad,
        grads_with_grad,
        grad_ests_with_grad,
        var_ests_with_grad,
        max_var_ests_with_grad,
        innov_res_with_grad,
        steps_with_grad,
    )


def _advance_psi(
    psi_1_prev: float,
    psi_2_prev: float,
    psi_3_prev: float,
    beta1: float,
    beta2: float,
    beta3: float,
) -> Tuple[float, float, float]:
    return (
        beta1 * psi_1_prev + 1.0,
        beta2 * psi_2_prev + 1.0,
        beta3 * psi_3_prev + 1.0,
    )


def _single_tensor_fused_iris(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    grad_estimates: List[torch.Tensor],
    variance_estimates: List[torch.Tensor],
    max_variance_estimates: List[torch.Tensor],
    innovation_residuals: List[torch.Tensor],
    state_steps: List[torch.Tensor],
    psi_1_prev: float,
    psi_2_prev: float,
    psi_3_prev: float,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    beta3: float,
    wd: float,
    eps: float,
    rho: Optional[float],
    amsgrad: bool,
    has_complex: bool,
) -> Tuple[float, float, float]:
    """fused=True, foreach=False: one iris_fused_cuda launch per tensor."""
    if not _fused_available:
        raise RuntimeError(
            "Fused IRIS kernel not available. Use foreach=True instead."
        )
    if has_complex:
        raise RuntimeError(
            "Fused IRIS kernel does not support complex parameters. Use foreach=True instead."
        )

    (
        params_with_grad,
        grads_with_grad,
        grad_ests_with_grad,
        var_ests_with_grad,
        max_var_ests_with_grad,
        innov_res_with_grad,
        steps_with_grad,
    ) = _collect_fused_tensors(
        params,
        grads,
        grad_estimates,
        variance_estimates,
        max_variance_estimates,
        innovation_residuals,
        state_steps,
        amsgrad,
    )

    if not params_with_grad:
        return _advance_psi(psi_1_prev, psi_2_prev, psi_3_prev, beta1, beta2, beta3)

    use_clipping = rho is not None
    rho_value = rho if rho is not None else 1.0
    psi_1_curr = psi_2_curr = psi_3_curr = None

    # Pass the same psi_*_prev into every call. iris_fused_cuda advances
    # psi internally; feeding the returned values back would apply the
    # bias-correction recurrence once per tensor instead of once per step.
    for i in range(len(params_with_grad)):
        max_var = (
            max_var_ests_with_grad[i] if amsgrad else var_ests_with_grad[i]
        )
        psi_1_curr, psi_2_curr, psi_3_curr = _fused_kernel.iris_fused_cuda(
            params_with_grad[i],
            grads_with_grad[i],
            grad_ests_with_grad[i],
            var_ests_with_grad[i],
            max_var,
            innov_res_with_grad[i],
            steps_with_grad[i],
            psi_1_prev,
            psi_2_prev,
            psi_3_prev,
            lr,
            beta1,
            beta2,
            beta3,
            wd,
            eps,
            rho_value,
            use_clipping,
            amsgrad,
        )

    return psi_1_curr, psi_2_curr, psi_3_curr


def _multi_tensor_fused_iris(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    grad_estimates: List[torch.Tensor],
    variance_estimates: List[torch.Tensor],
    max_variance_estimates: List[torch.Tensor],
    innovation_residuals: List[torch.Tensor],
    state_steps: List[torch.Tensor],
    psi_1_prev: float,
    psi_2_prev: float,
    psi_3_prev: float,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    beta3: float,
    wd: float,
    eps: float,
    rho: Optional[float],
    amsgrad: bool,
    has_complex: bool,
) -> Tuple[float, float, float]:
    """fused=True, foreach=True: one iris_multi_tensor_fused_cuda launch."""
    if not _fused_available:
        raise RuntimeError(
            "Fused IRIS kernel not available. Use foreach=True instead."
        )
    if has_complex:
        raise RuntimeError(
            "Fused IRIS kernel does not support complex parameters. Use foreach=True instead."
        )

    (
        params_with_grad,
        grads_with_grad,
        grad_ests_with_grad,
        var_ests_with_grad,
        max_var_ests_with_grad,
        innov_res_with_grad,
        steps_with_grad,
    ) = _collect_fused_tensors(
        params,
        grads,
        grad_estimates,
        variance_estimates,
        max_variance_estimates,
        innovation_residuals,
        state_steps,
        amsgrad,
    )

    if not params_with_grad:
        return _advance_psi(psi_1_prev, psi_2_prev, psi_3_prev, beta1, beta2, beta3)

    use_clipping = rho is not None
    rho_value = rho if rho is not None else 1.0

    return _fused_kernel.iris_multi_tensor_fused_cuda(
        params_with_grad,
        grads_with_grad,
        grad_ests_with_grad,
        var_ests_with_grad,
        max_var_ests_with_grad if amsgrad else [],
        innov_res_with_grad,
        steps_with_grad,
        psi_1_prev,
        psi_2_prev,
        psi_3_prev,
        lr,
        beta1,
        beta2,
        beta3,
        wd,
        eps,
        rho_value,
        use_clipping,
        amsgrad,
    )


def compile_cuda_extension():
    """
    Compile the IRIS fused CUDA extension.
    
    Attempts to build the CUDA kernel from source.
    If successful, the fused kernel will be available.
    """
    try:
        from torch.utils.cpp_extension import load
        import os
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        fused_dir = os.path.join(current_dir, "fused_iris")
        sources = [
            os.path.join(fused_dir, "pybind_iris.cpp"),
            os.path.join(fused_dir, "fused_iris_kernel.cu"),
            os.path.join(fused_dir, "multi_tensor_iris_kernel.cu"),
        ]
        missing = [p for p in sources if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"CUDA kernel source not found: {missing}"
            )
        
        print("Building IRIS fused CUDA kernel...")
        
        fused_iris = load(
            name="fused_iris_kernel",
            sources=sources,
            extra_include_paths=[fused_dir],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=True
        )
        
        print("✓ IRIS fused kernel built successfully")
        return fused_iris
        
    except Exception as e:
        print(f"✗ Failed to build IRIS fused kernel: {e}")
        print("You can still use IRIS with foreach=True")
        raise


__all__ = ["is_fused_available", "compile_cuda_extension"]