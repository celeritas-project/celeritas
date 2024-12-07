//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/detail/TrackingCutExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/Types.hh"
#include "celeritas/optical/CoreTrackView.hh"
#include "celeritas/optical/ParticleTrackView.hh"
#include "celeritas/optical/SimTrackView.hh"

namespace celeritas
{
namespace optical
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Kill a malfunctioning photon.
 */
struct TrackingCutExecutor
{
    inline CELER_FUNCTION void operator()(CoreTrackView& track);
};

//---------------------------------------------------------------------------//
CELER_FUNCTION void TrackingCutExecutor::operator()(CoreTrackView& track)
{
    using Energy = ParticleTrackView::Energy;

    auto deposited = track.particle().energy().value();

    auto&& sim = track.sim();
#if !CELER_DEVICE_COMPILE
    {
        // Print a debug message if track is just being cut; error message if
        // an error occurred
        auto msg = self_logger()(CELER_CODE_PROVENANCE,
                                 sim.status() == TrackStatus::errored
                                     ? LogLevel::error
                                     : LogLevel::debug);
        msg << "Killing optical photon: lost " << deposited << ' '
            << Energy::unit_type::label();
    }
#endif

    sim.status(TrackStatus::killed);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
