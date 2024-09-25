//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/TrackInitParams.cc
//---------------------------------------------------------------------------//
#include "TrackInitParams.hh"

#include <utility>

#include "corecel/Assert.hh"

#include "TrackInitData.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct with capacity.
 */
TrackInitParams::TrackInitParams(size_type capacity)
{
    CELER_EXPECT(capacity > 0);

    HostVal<TrackInitParamsData> host_data;
    host_data.capacity = capacity;
    CELER_ASSERT(host_data);
    data_ = CollectionMirror<TrackInitParamsData>{std::move(host_data)};
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
