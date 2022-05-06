//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/launcher/MollerBhabhaLauncher.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/global/CoreTrackView.hh"

#include "celeritas/em/data/MollerBhabhaData.hh"
#include "celeritas/em/interactor/MollerBhabhaInteractor.hh"

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
