//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/InteractionApplier.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/cont/Span.hh"
#include "corecel/sys/KernelTraits.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/geo/GeoFwd.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/track/SimTrackView.hh"

#include "CutoffView.hh"
#include "Interaction.hh"
#include "ParticleTrackView.hh"
#include "ParticleView.hh"
#include "PhysicsData.hh"
#include "PhysicsStepView.hh"
#include "PhysicsTrackView.hh"
#include "Secondary.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Wrap an Interaction executor to apply it to a track.
 *
 * The function F must take a \c CoreTrackView and return a \c Interaction
 */
template<class F>
struct InteractionApplierBaseImpl
{
    //// DATA ////

    F sample_interaction;

    //// METHODS ////

    CELER_FUNCTION void operator()(celeritas::CoreTrackView const&);
};

//---------------------------------------------------------------------------//
/*!
 *
 * This class is partially specialized with a second template argument to
 * extract any launch bounds from the functor class. TODO: we could probably
 * inherit from a helper class to pull in those constants (if available).
 */
template<class F, typename = void>
struct InteractionApplier : public InteractionApplierBaseImpl<F>
{
    CELER_FUNCTION InteractionApplier(F&& f)
        : InteractionApplierBaseImpl<F>{celeritas::forward<F>(f)}
    {
    }
};

template<class F>
struct InteractionApplier<F, std::enable_if_t<kernel_max_blocks_min_warps<F>>>
    : public InteractionApplierBaseImpl<F>
{
    static constexpr int max_block_size = F::max_block_size;
    static constexpr int min_warps_per_eu = F::min_warps_per_eu;

    CELER_FUNCTION InteractionApplier(F&& f)
        : InteractionApplierBaseImpl<F>{celeritas::forward<F>(f)}
    {
    }
};

template<class F>
struct InteractionApplier<F, std::enable_if_t<kernel_max_blocks<F>>>
    : public InteractionApplierBaseImpl<F>
{
    static constexpr int max_block_size = F::max_block_size;

    CELER_FUNCTION InteractionApplier(F&& f)
        : InteractionApplierBaseImpl<F>{celeritas::forward<F>(f)}
    {
    }
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class F>
CELER_FUNCTION InteractionApplier(F&&) -> InteractionApplier<F>;

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
InteractionApplierBaseImpl<F>::operator()(celeritas::CoreTrackView const& track)
{
    Interaction result = this->sample_interaction(track);

    auto sim = track.sim();
    if (CELER_UNLIKELY(result.action == Interaction::Action::failed))
    {
        auto phys = track.physics();
        // Particle already moved to the collision site, but an out-of-memory
        // (allocation failure) occurred. Someday we can add error handling,
        // but for now use the "failure" action in the physics and set the step
        // limit to zero since it needs to interact again at this location.
        sim.step_limit({0, phys.scalars().failure_action()});
        return;
    }
    else if (!result.changed())
    {
        return;
    }

    // Scattered or absorbed
    {
        // Update post-step energy
        auto particle = track.particle();
        particle.energy(result.energy);
    }

    if (result.action != Interaction::Action::absorbed)
    {
        // Update direction
        auto geo = track.geometry();
        geo.set_dir(result.direction);
    }
    else
    {
        // Mark particle as dead
        sim.status(TrackStatus::killed);
    }

    real_type deposition = result.energy_deposition.value();
    auto cutoff = track.cutoff();
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
                deposition += secondary.energy.value() * sim.weight();
                auto sec_par = track.particle_record(secondary.particle_id);
                if (sec_par.is_antiparticle())
                {
                    // Conservation of energy for positrons
                    deposition += 2 * sec_par.mass().value();
                }
                secondary = {};
            }
        }
    }
    auto phys = track.physics_step();
    phys.deposit_energy(units::MevEnergy{deposition});
    phys.secondaries(result.secondaries);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
