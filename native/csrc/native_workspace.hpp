#pragma once

#include <cstddef>
#include <cstdint>

#include "thermal_2s_impl.hpp"

namespace py2sess_native {

inline std::size_t rt_workspace_bytes(std::int64_t nlay) {
  const auto thermal_bytes = thermal_2s_workspace_bytes<double>(nlay);
  const auto solar_bytes = solar_2s_workspace_bytes<double>(nlay);
  return thermal_bytes > solar_bytes ? thermal_bytes : solar_bytes;
}

}  // namespace py2sess_native
