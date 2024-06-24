//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/detail/GeoErrorExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "celeritas/Types.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/track/SimTrackView.hh"

#include "../GeoMaterialView.hh"
#include "../GeoTrackView.hh"

#if !CELER_DEVICE_COMPILE
#    include "corecel/io/Logger.hh"
#    include "corecel/io/Repr.hh"
#endif

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Kill the track due to a geometry error.
 */
struct GeoErrorExecutor
{
    inline CELER_FUNCTION void
    operator()(celeritas::CoreTrackView const& track);
};

//---------------------------------------------------------------------------//
CELER_FUNCTION void
GeoErrorExecutor::operator()(celeritas::CoreTrackView const& track)
{
    using Energy = ParticleTrackView::Energy;

    auto particle = track.make_particle_view();
    auto sim = track.make_sim_view();

    // If the track is looping (or if it's a stuck track that was
    // flagged as looping), deposit the energy locally.
    auto deposited = particle.energy().value();
    if (particle.is_antiparticle())
    {
        // Energy conservation for killed positrons
        deposited += 2 * particle.mass().value();
    }
    track.make_physics_step_view().deposit_energy(Energy{deposited});
    particle.subtract_energy(particle.energy());

    // Mark that this track was abandoned while looping
    sim.status(TrackStatus::killed);

#if !CELER_DEVICE_COMPILE
    auto geo = track.make_geo_view();
    auto msg = CELER_LOG(error);
    msg << "Tracking error at " << repr(geo.pos()) << " along "
        << repr(geo.dir()) << ": depositing " << deposited << " ["
        << Energy::unit_type::label() << "] in ";
    if (geo.is_outside())
    {
        msg << "exterior";
    }
    else
    {
        msg << "volume " << geo.volume_id().unchecked_get();
    }
#endif
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
