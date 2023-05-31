//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/InteractionApplier.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/cont/Span.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/geo/GeoFwd.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/phys/CutoffView.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/phys/ParticleView.hh"
#include "celeritas/phys/PhysicsData.hh"
#include "celeritas/phys/PhysicsStepView.hh"
#include "celeritas/phys/PhysicsTrackView.hh"
#include "celeritas/phys/Secondary.hh"
#include "celeritas/track/SimTrackView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Wrap an Interaction executor to apply it to a track.
 *
 * The function F must take a \c CoreTrackView and return a \c Interaction
 */
template<class F>
struct InteractionApplier
{
    //// DATA ////

    F sample_interaction;

    //// METHODS ////

    CELER_FUNCTION void operator()(celeritas::CoreTrackView const&);
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class F>
CELER_FUNCTION InteractionApplier(F&&)->InteractionApplier<F>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Sample an interaction and apply to the track view.
 *
 * The given track *must* be an active track with the correct step limit action
 * ID.
 */
template<class F>
CELER_FUNCTION void
InteractionApplier<F>::operator()(celeritas::CoreTrackView const& track)
{
    Interaction result = this->sample_interaction(track);

    if (!result.changed())
    {
        return;
    }

    auto sim = track.make_sim_view();
    if (CELER_UNLIKELY(result.action == Interaction::Action::failed))
    {
        auto phys = track.make_physics_view();
        // Particle already moved to the collision site, but an out-of-memory
        // (allocation failure) occurred. Someday we can add error handling,
        // but for now use the "failure" action in the physics and set the step
        // limit to zero since it needs to interact again at this location.
        sim.step_limit({0, phys.scalars().failure_action()});
        return;
    }

    // Scattered or absorbed
    {
        // Update post-step energy
        auto particle = track.make_particle_view();
        particle.energy(result.energy);
    }

    if (result.action != Interaction::Action::absorbed)
    {
        // Update direction
        auto geo = track.make_geo_view();
        geo.set_dir(result.direction);
    }
    else
    {
        // Mark particle as dead
        sim.status(TrackStatus::killed);
    }

    real_type deposition = result.energy_deposition.value();
    auto cutoff = track.make_cutoff_view();
    if (cutoff.apply_post_interaction())
    {
        // Kill secondaries with energies below the production cut
        for (auto& secondary : result.secondaries)
        {
            if (cutoff.apply(secondary))
            {
                // Secondary is an electron, positron or gamma with energy
                // below the production cut -- deposit the energy locally
                // and clear the secondary
                deposition += secondary.energy.value();
                auto sec_par = track.make_particle_view(secondary.particle_id);
                if (sec_par.is_antiparticle())
                {
                    // Conservation of energy for positrons
                    deposition += 2 * sec_par.mass().value();
                }
                secondary = {};
            }
        }
    }
    auto phys = track.make_physics_step_view();
    phys.deposit_energy(units::MevEnergy{deposition});
    phys.secondaries(result.secondaries);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
