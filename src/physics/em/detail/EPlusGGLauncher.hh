//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EPlusGGLauncher.hh
//---------------------------------------------------------------------------//
#pragma once

#include "sim/CoreTrackView.hh"

#include "EPlusGGData.hh"
#include "EPlusGGInteractor.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Apply the EPlusGGInteractor to the current track.
 */
inline CELER_FUNCTION Interaction
eplusgg_interact_track(EPlusGGData const& model, CoreTrackView const& track)
{
    auto        allocate_secondaries = track.make_secondary_allocator();
    auto        particle             = track.make_particle_view();
    const auto& dir                  = track.make_geo_view().dir();

    EPlusGGInteractor interact(model, particle, dir, allocate_secondaries);
    auto              rng = track.make_rng_engine();
    return interact(rng);
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
