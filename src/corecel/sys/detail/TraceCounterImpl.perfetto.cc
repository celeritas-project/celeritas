//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/detail/TraceCounterImpl.perfetto.cc
//! \brief Numeric tracing counter
//---------------------------------------------------------------------------//

#include "TraceCounterImpl.hh"

#include <type_traits>
#include <perfetto.h>

#include "TrackEvent.perfetto.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//! Implement NVTX counters for Perfetto
template<class T>
void trace_counter_impl(char const* name, T value)
{
    static_assert(std::is_arithmetic_v<T>, "Only support numeric counters");
    TRACE_COUNTER(detail::perfetto_track_event_category,
                  ::perfetto::DynamicString{name},
                  value);
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATIONS
//---------------------------------------------------------------------------//

template void trace_counter_impl(char const*, double);
template void trace_counter_impl(char const*, float);
template void trace_counter_impl(char const*, std::int32_t);
template void trace_counter_impl(char const*, std::int64_t);
template void trace_counter_impl(char const*, std::uint32_t);
template void trace_counter_impl(char const*, std::uint64_t);

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
