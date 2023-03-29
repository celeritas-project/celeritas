//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/detail/InteractionLauncherImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/global/CoreTrackView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Function-like class to launch a "Track"-dependent function from core and
 * model data.
 *
 * This class should be used by generated interactor functions.
 */
template<class D, class F>
struct InteractionLauncherImpl
{
    //!@{
    //! \name Type aliases
    using CoreRefNative = CoreRef<MemSpace::native>;
    //!@}

    //// DATA ////

    CoreRefNative const& core_data;
    D const& model_data;
    F call_with_track;

    //// METHODS ////

    CELER_FUNCTION void operator()(ThreadId thread) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Apply the interaction to the track with the given thread ID.
 */
template<class D, class F>
CELER_FUNCTION void
InteractionLauncherImpl<D, F>::operator()(ThreadId thread) const
{
    CELER_ASSERT(thread < this->core_data.states.size());
    const celeritas::CoreTrackView track(
        this->core_data.params, this->core_data.states, thread);

    auto sim = track.make_sim_view();
    if (sim.step_limit().action != model_data.ids.action)
        return;

    Interaction result = this->call_with_track(model_data, track);

    if (result.changed())
    {
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
        if (cutoff.apply_cuts())
        {
            // Kill secondaries with energies below the production cut
            for (auto& secondary : result.secondaries)
            {
                if (cutoff.apply_cut(secondary))
                {
                    // Secondary is an electron, positron or gamma with energy
                    // below the production cut -- deposit the energy locally
                    // and kill it
                    deposition += secondary.energy.value();
                    ParticleView particle{this->core_data.params.particles,
                                          secondary.particle_id};
                    if (particle.is_antiparticle())
                    {
                        // Conservation of energy for positrons
                        deposition += 2 * particle.mass().value();
                    }
                    secondary = {};
                }
            }
        }
        auto phys = track.make_physics_step_view();
        phys.deposit_energy(units::MevEnergy{deposition});
        phys.secondaries(result.secondaries);
    }
    else if (CELER_UNLIKELY(result.action == Interaction::Action::failed))
    {
        auto phys = track.make_physics_view();
        // Particle already moved to the collision site, but an out-of-memory
        // (allocation failure) occurred. Someday we can add error handling,
        // but for now use the "failure" action in the physics and set the step
        // limit to zero since it needs to interact again at this location.
        sim.step_limit({0, phys.scalars().failure_action()});
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
