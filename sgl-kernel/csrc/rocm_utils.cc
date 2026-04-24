/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/**
 * ROCm runtime utility functions for multi-architecture support.
 *
 * These functions must be defined in a .cc file (not inline in a header)
 * to avoid ODR (One Definition Rule) violations when building shared libraries.
 * Static variables in inline functions can cause multiple definitions across
 * compilation units, leading to incorrect behavior in wheel distributions.
 */

#ifdef USE_ROCM

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

namespace sgl_kernel {

enum class FP8TypeEnum {
  E4M3FNUZ,  // For gfx942 (MI300/MI325)
  E4M3FN     // For gfx950 (MI350)
};

// Get base GPU architecture name (e.g., "gfx942" or "gfx950")
std::string get_device_arch_name() {
  static std::string cached_arch;
  if (!cached_arch.empty()) {
    return cached_arch;
  }

  int device = -1;
  hipError_t err = hipGetDevice(&device);
  if (err != hipSuccess) {
    throw std::runtime_error("Failed to get current device");
  }

  hipDeviceProp_t prop;
  err = hipGetDeviceProperties(&prop, device);
  if (err != hipSuccess) {
    throw std::runtime_error("Failed to get device properties");
  }

  // Extract base architecture (e.g., "gfx942:sramecc+:xnack-" -> "gfx942")
  std::string full_arch(prop.gcnArchName);
  size_t colon_pos = full_arch.find(':');
  cached_arch = (colon_pos != std::string::npos) ? full_arch.substr(0, colon_pos) : full_arch;

  return cached_arch;
}

// Get FP8 type for current GPU
FP8TypeEnum get_fp8_type() {
  static FP8TypeEnum cached_type = FP8TypeEnum::E4M3FN;
  static bool initialized = false;

  if (!initialized) {
    std::string arch = get_device_arch_name();
    if (arch == "gfx942") {
      cached_type = FP8TypeEnum::E4M3FNUZ;
    } else if (arch == "gfx950") {
      cached_type = FP8TypeEnum::E4M3FN;
    } else {
      throw std::runtime_error("Unsupported GPU architecture for FP8: " + arch);
    }
    initialized = true;
  }

  return cached_type;
}

// Get TopK dynamic shared memory size for current GPU
size_t get_topk_smem_size() {
  static size_t cached_smem = 0;
  if (cached_smem != 0) {
    return cached_smem;
  }

  std::string arch = get_device_arch_name();
  if (arch == "gfx942") {
    // gfx942 (MI300/MI325): LDS is typically 64KB per workgroup
    // Keep dynamic smem <= ~48KB (leaves room for static shared allocations)
    cached_smem = 48 * 1024;
  } else if (arch == "gfx950") {
    // gfx950 (MI350): LDS is larger (e.g. 160KB per CU)
    // Allow the original 128KB dynamic smem
    cached_smem = 128 * 1024;
  } else {
    throw std::runtime_error("Unsupported GPU architecture for TopK SMEM: " + arch);
  }

  return cached_smem;
}

}  // namespace sgl_kernel

#endif  // USE_ROCM
