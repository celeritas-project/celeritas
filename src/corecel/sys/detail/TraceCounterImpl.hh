//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/detail/TraceCounterImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstdint>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
template<class T>
void trace_counter_impl(char const* name, T value);

#if CELERITAS_USE_PERFETTO || CELERITAS_USE_CUDA

// On some platform size_t is equivalent to uint64_t, which would cause
// duplicate template instantiation
template<class T>
using trace_counter_type = std::conditional_t<
    std::is_same_v<T, std::size_t>,
    std::conditional_t<sizeof(std::size_t) == sizeof(std::uint64_t),
                       std::uint64_t,
                       std::uint32_t>,
    T>;

// Explicit instantiations
extern template void trace_counter_impl(char const*, double);
extern template void trace_counter_impl(char const*, float);
extern template void trace_counter_impl(char const*, std::int32_t);
extern template void trace_counter_impl(char const*, std::int64_t);
extern template void trace_counter_impl(char const*, std::uint32_t);
extern template void trace_counter_impl(char const*, std::uint64_t);

#else

template<class T>
using trace_counter_type = T;

template<class T>
inline void trace_counter_impl(char const*, T)
{
    // CUDA/PERFETTO are disabled
    CELER_ASSERT_UNREACHABLE();
}

#endif

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
