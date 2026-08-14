#pragma once

#include <ATen/ATen.h>

// Apex / Adan dispatch macros. Used instead of AT_DISPATCH when the .cu
// files need a named scalar_t_0 (same convention as sail-sg/Adan).

#define DISPATCH_FLOAT_AND_HALF(TYPE, LEVEL, NAME, ...) \
  switch (TYPE) { \
    case at::ScalarType::Float: { \
      using scalar_t_##LEVEL = float; \
      __VA_ARGS__; \
      break; \
    } \
    case at::ScalarType::Half: { \
      using scalar_t_##LEVEL = at::Half; \
      __VA_ARGS__; \
      break; \
    } \
    default: \
      AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'"); \
  }

#define DISPATCH_DOUBLE_AND_FLOAT(TYPE, LEVEL, NAME, ...) \
  switch (TYPE) { \
    case at::ScalarType::Double: { \
      using scalar_t_##LEVEL = double; \
      __VA_ARGS__; \
      break; \
    } \
    case at::ScalarType::Float: { \
      using scalar_t_##LEVEL = float; \
      __VA_ARGS__; \
      break; \
    } \
    default: \
      AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'"); \
  }

#define DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT(TYPE, LEVEL, NAME, ...) \
  switch (TYPE) { \
    case at::ScalarType::Double: { \
      using scalar_t_##LEVEL = double; \
      __VA_ARGS__; \
      break; \
    } \
    case at::ScalarType::Float: { \
      using scalar_t_##LEVEL = float; \
      __VA_ARGS__; \
      break; \
    } \
    case at::ScalarType::Half: { \
      using scalar_t_##LEVEL = at::Half; \
      __VA_ARGS__; \
      break; \
    } \
    case at::ScalarType::BFloat16: { \
      using scalar_t_##LEVEL = at::BFloat16; \
      __VA_ARGS__; \
      break; \
    } \
    default: \
      AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'"); \
  }
