//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MollerBhabhaLauncher.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Assert.hh"
#include "base/Macros.hh"
#include "sim/CoreTrackView.hh"

#include "MollerBhabhaData.hh"
#include "MollerBhabhaInteractor.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Apply MollerBhabha to the current track.
 */
inline CELER_FUNCTION Interaction moller_bhabha_interact_track(
    MollerBhabhaData const& model, CoreTrackView const& track)
{
    auto        particle             = track.make_particle_view();
    auto        cutoff               = track.make_cutoff_view();
    const auto& dir                  = track.make_geo_view().dir();
    auto        allocate_secondaries = track.make_secondary_allocator();

    MollerBhabhaInteractor interact(
        model, particle, cutoff, dir, allocate_secondaries);
    auto rng = track.make_rng_engine();
    return interact(rng);
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
