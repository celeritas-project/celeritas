//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/DynamicGeoData.cc
//---------------------------------------------------------------------------//
#include "DynamicGeoData.hh"

#include "corecel/Assert.hh"
#include "geocel/WrappedGeoTrackView.hh"
#include "orange/OrangeTrackView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Create a reusable implementation bound to the given params and state.
 *
 * The implementation keeps copies of the (lightweight) data references, so
 * the arguments need not outlive it, but the underlying data must.
 *
 * \todo Select the implementation at runtime rather than hardcoding ORANGE.
 */
std::shared_ptr<DynamicGeoTrackInterface> make_dynamic_geo_track(
    HostCRef<OrangeParamsData> const& params,
    HostRef<OrangeStateData> const& states)
{
    CELER_EXPECT(params);
    CELER_EXPECT(states);
    return std::make_shared<WrappedGeoTrackView<OrangeTrackView>>(
        params, states, TrackSlotId{0});
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
