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
#    include "celeritas/global/Debug.hh"
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
    inline CELER_FUNCTION void operator()(celeritas::CoreTrackView& track);
};

//---------------------------------------------------------------------------//
CELER_FUNCTION void
TrackingCutExecutor::operator()(celeritas::CoreTrackView& track)
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

#if !CELER_DEVICE_COMPILE
    {
        // Print a debug message if track is just being cut; error message if
        // an error occurred
        auto const geo = track.make_geo_view();
        auto msg = self_logger()(CELER_CODE_PROVENANCE,
                                 sim.status() == TrackStatus::errored
                                     ? LogLevel::error
                                     : LogLevel::debug);
        msg << "Killing track " << StreamableTrack{track} << ": "
            << (geo.is_outside() ? "depositing" : "lost") << ' ' << deposited
            << ' ' << Energy::unit_type::label();
    }
#endif

    track.make_physics_step_view().deposit_energy(Energy{deposited});
    particle.subtract_energy(particle.energy());

    sim.status(TrackStatus::killed);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
