//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/TrackInitParams.cc
//---------------------------------------------------------------------------//
#include "TrackInitParams.hh"

#include <utility>

#include "corecel/Assert.hh"
#include "corecel/data/Filler.hh"

#include "TrackInitData.hh"  // IWYU pragma: associated

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
    CELER_EXPECT(inp.track_order < TrackOrder::size_);

    HostVal<TrackInitParamsData> host_data;
    host_data.capacity = inp.capacity;
    host_data.max_events = inp.max_events;
    host_data.track_order = inp.track_order;
    CELER_ASSERT(host_data);
    data_ = ParamsDataStore<TrackInitParamsData>{std::move(host_data)};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
