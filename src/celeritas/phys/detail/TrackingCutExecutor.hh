//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/detail/TrackingCutExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "celeritas/Types.hh"
#include "celeritas/geo/GeoTrackView.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/track/SimTrackView.hh"

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
 * Kill the current track and deposit its energy.
 *
 * This is called to kill a track due to "user cuts" (i.e., minimum energy,
 * maximum number of steps, maximum lab-frame time) and due to geometry errors
 * (i.e.  initialization, boundary crossing). It deposits the track's energy
 * plus, if an anitiparticle, the annihilation energy as well.
 *
 * If the track has an "error" status and the track is on the host, a message
 * will be printed.
 *
 * TODO: we could use a stack allocator to perform a reduction in this kernel
 * so that the host can print out warning messages (or print at the end of the
 * simulation).
 */
struct TrackingCutExecutor
{
    inline CELER_FUNCTION void
    operator()(celeritas::CoreTrackView const& track);
};

//---------------------------------------------------------------------------//
CELER_FUNCTION void
TrackingCutExecutor::operator()(celeritas::CoreTrackView const& track)
{
    using Energy = ParticleTrackView::Energy;

    auto particle = track.make_particle_view();
    auto sim = track.make_sim_view();

    // Deposit the remaining energy locally
    auto deposited = particle.energy().value();
    if (particle.is_antiparticle())
    {
        // Energy conservation for killed positrons
        deposited += 2 * particle.mass().value();
    }
    track.make_physics_step_view().deposit_energy(Energy{deposited});
    particle.subtract_energy(particle.energy());

#if !CELER_DEVICE_COMPILE
    if (sim.status() == TrackStatus::errored)
    {
        auto geo = track.make_geo_view();
        auto msg = CELER_LOG_LOCAL(error);
        msg << "Tracking error (track ID " << sim.track_id().get()
            << ", track slot " << track.track_slot_id().get() << ") at "
            << repr(geo.pos()) << " along " << repr(geo.dir()) << ": ";
        if (!geo.is_outside())
        {
            msg << "depositing " << deposited << " ["
                << Energy::unit_type::label() << "] in "
                << "volume " << geo.volume_id().unchecked_get();
        }
        else
        {
            msg << "lost " << deposited << " " << Energy::unit_type::label()
                << " energy";
        }
    }
#endif

    sim.status(TrackStatus::killed);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
