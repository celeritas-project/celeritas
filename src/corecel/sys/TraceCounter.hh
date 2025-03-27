//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/TraceCounter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <cstdint>

#include "corecel/Config.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// Simple tracing counter
template<class T>
void trace_counter(char const* name, T value);

#if CELERITAS_USE_PERFETTO || CELERITAS_USE_CUDA
// Explicit instantiations
extern template void trace_counter(char const*, std::uint32_t);
extern template void trace_counter(char const*, std::uint64_t);
extern template void trace_counter(char const*, std::int32_t);
extern template void trace_counter(char const*, std::int64_t);
extern template void trace_counter(char const*, float);
extern template void trace_counter(char const*, double);

#else

// Ignore if Perfetto is unavailable
template<class T>
inline void trace_counter(char const*, T)
{
}

#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
