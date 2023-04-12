//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/TrackInitParams.cc
//---------------------------------------------------------------------------//
#include "TrackInitParams.hh"

#include <utility>

#include "corecel/Assert.hh"
#include "celeritas/track/TrackInitData.hh"  // IWYU pragma: associated

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with capacity and number of events.
 */
TrackInitParams::TrackInitParams(Input const& inp)
{
    CELER_EXPECT(inp.capacity > 0);
    CELER_EXPECT(inp.max_events > 0);

    HostVal<TrackInitParamsData> host_data;
    host_data.capacity = inp.capacity;
    host_data.max_events = inp.max_events;
    host_data.track_order = inp.track_order;
    CELER_ASSERT(host_data);
    data_ = CollectionMirror<TrackInitParamsData>{std::move(host_data)};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
