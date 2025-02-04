//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/TraceCounter.perfetto.cc
//! \brief Numeric tracing counter
//---------------------------------------------------------------------------//
#include "TraceCounter.hh"

#include <type_traits>
#include <perfetto.h>

#include "corecel/Types.hh"

#include "detail/TrackEvent.perfetto.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Simple performance tracing counter.
 * \tparam T Arithmetic counter type
 *
 * Records a named value at the current timestamp which
 * can then be displayed on a timeline. Only supported on host, this compiles
 * but is a noop on device.
 *
 * See https://perfetto.dev/docs/instrumentation/track-events#counters
 */
template<class T>
void trace_counter(char const* name, T value)
{
    static_assert(std::is_arithmetic_v<T>, "Only support numeric counters");
    TRACE_COUNTER(detail::perfetto_track_event_category,
                  ::perfetto::DynamicString{name},
                  value);
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATIONS
//---------------------------------------------------------------------------//

template void trace_counter(char const*, unsigned int);
template void trace_counter(char const*, std::size_t);
template void trace_counter(char const*, float);
template void trace_counter(char const*, double);

//---------------------------------------------------------------------------//
}  // namespace celeritas
