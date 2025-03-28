//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/TraceCounter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "corecel/Config.hh"

#include "ScopedProfiling.hh"
#if CELERITAS_USE_PERFETTO || CELERITAS_USE_CUDA
#    include "detail/TraceCounterImpl.hh"
#endif

namespace celeritas
{
#if CELERITAS_USE_PERFETTO || CELERITAS_USE_CUDA
//---------------------------------------------------------------------------//
// Simple tracing counter
template<class T>
inline void trace_counter(char const* name, T value)
{
    static_assert(std::is_arithmetic_v<T>, "Only support numeric counters");
    if (use_profiling())
    {
        // on some platform size_t is equivalent to uint64_t, which would cause
        // duplicate template instantiation
        using counter_type = std::conditional_t<
            std::is_same_v<T, std::size_t>,
            std::conditional_t<sizeof(std::size_t) == sizeof(std::uint64_t),
                               std::uint64_t,
                               std::uint32_t>,
            T>;
        detail::trace_counter_impl<counter_type>(name, value);
    }
}
#else

// Ignore if Perfetto is unavailable
template<class T>
inline void trace_counter(char const*, T)
{
}

#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
