//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/Counter.perfetto.cc
//! \brief Numeric tracing counter
//---------------------------------------------------------------------------//

#include "Counter.hh"

#include <type_traits>
#include <perfetto.h>

#include "corecel/Types.hh"

#include "detail/TrackEvent.perfetto.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
template<class T>
void trace_counter(char const* name, T value)
{
    static_assert(std::is_arithmetic_v<T>, "Only support numeric counters");
    TRACE_COUNTER(detail::perfetto_track_event_category, name, value);
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATIONS
//---------------------------------------------------------------------------//

template void trace_counter(char const*, size_type);
template void trace_counter(char const*, float);
template void trace_counter(char const*, double);

//---------------------------------------------------------------------------//
}  // namespace celeritas