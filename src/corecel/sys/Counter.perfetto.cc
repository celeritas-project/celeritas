//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/Counter.perfetto.cc
//! \brief Numeric tracing counter
//---------------------------------------------------------------------------//

#include "Counter.hh"

#include <corecel/Types.hh>
#include <perfetto.h>

#include "detail/TrackEvent.perfetto.hh"

namespace celeritas
{
template<class T>
Counter<T>::Counter(char const* name, T value)
{
    TRACE_COUNTER(detail::perfetto_track_event_category, name, value);
}

template class Counter<size_type>;
template class Counter<float>;
template class Counter<double>;

}  // namespace celeritas