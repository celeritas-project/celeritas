//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/launcher/KleinNishinaLauncher.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/em/data/KleinNishinaData.hh"
#include "celeritas/em/interactor/KleinNishinaInteractor.hh"
#include "celeritas/global/CoreTrackView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Apply the KleinNishinaInteractor to the current track.
 */
inline CELER_FUNCTION Interaction klein_nishina_interact_track(
    KleinNishinaData const& model, CoreTrackView const& track)
{
    auto        allocate_secondaries = track.make_secondary_allocator();
    auto        particle             = track.make_particle_view();
    const auto& dir                  = track.make_geo_view().dir();

    KleinNishinaInteractor interact(model, particle, dir, allocate_secondaries);
    auto                   rng = track.make_rng_engine();
    return interact(rng);
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
