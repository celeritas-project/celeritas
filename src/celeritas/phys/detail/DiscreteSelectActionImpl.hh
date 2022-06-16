//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/detail/DiscreteSelectActionImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/global/CoreTrackView.hh"

#include "../PhysicsStepUtils.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Select a physics process before undergoing a collision.
 */
inline CELER_FUNCTION void
discrete_select_track(celeritas::CoreTrackView const& track)
{
    auto sim = track.make_sim_view();
    if (sim.status() != TrackStatus::alive)
    {
        // TODO: this check should be redundant with the action check below,
        // but we currently need to create the physics track view to check the
        // scalars (and this cannot be done if the track is inactive).
        return;
    }

    auto phys = track.make_physics_view();
    if (sim.step_limit().action != phys.scalars().discrete_action())
    {
        // This kernel does not apply
        return;
    }

    // Reset the MFP counter, to be resampled if the track survives the
    // interaction
    phys.reset_interaction_mfp();

    auto particle = track.make_particle_view();

    // Select the action to take
    auto rng      = track.make_rng_engine();
    auto step     = track.make_physics_step_view();
    auto selected = select_discrete_interaction(particle, phys, step, rng);

    // Save it as the next kernel
    sim.force_step_limit(selected.action);

    // Sample an element for models that use precalculated elemental cross
    // sections
    if (selected.action != phys.scalars().integral_rejection_action())
    {
        auto material = track.make_material_view().make_material_view();
        auto elcomp_id
            = phys.sample_element(material,
                                  selected.process,
                                  phys.action_to_model(selected.action),
                                  particle.energy(),
                                  rng);
        phys.element_id(elcomp_id);
    }

    CELER_ENSURE(!phys.has_interaction_mfp());
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
