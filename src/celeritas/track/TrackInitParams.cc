//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/TrackInitParams.cc
//---------------------------------------------------------------------------//
#include "TrackInitParams.hh"

#include "corecel/Assert.hh"
#include "celeritas/track/TrackInitData.hh" // IWYU pragma: associated

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with capacity and number of events.
 */
TrackInitParams::TrackInitParams(const Input& inp)
{
    CELER_EXPECT(inp.capacity > 0);
    CELER_EXPECT(inp.max_events > 0);

    HostVal<TrackInitParamsData> host_data;
    host_data.capacity   = inp.capacity;
    host_data.max_events = inp.max_events;
    CELER_ASSERT(host_data);
    data_ = CollectionMirror<TrackInitParamsData>{std::move(host_data)};
}

//---------------------------------------------------------------------------//
} // namespace celeritas
