//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RayleighLauncher.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Assert.hh"
#include "base/Types.hh"
#include "random/RngEngine.hh"
#include "physics/base/ModelData.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/material/Types.hh"
#include "physics/material/MaterialTrackView.hh"
#include "physics/material/ElementView.hh"
#include "physics/material/ElementSelector.hh"
#include "physics/base/PhysicsTrackView.hh"
#include "RayleighInteractor.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Model interactor kernel launcher
 */
template<MemSpace M>
struct RayleighLauncher
{
    CELER_FUNCTION RayleighLauncher(const RayleighNativeRef&    pointers,
                                    const ModelInteractRefs<M>& interaction)
        : rayleigh(pointers), model(interaction)
    {
    }

    const RayleighNativeRef&    rayleigh; //!< Shared data for interactor
    const ModelInteractRefs<M>& model;    //!< State data needed to interact

    //! Create track views and launch interactor
    inline CELER_FUNCTION void operator()(ThreadId tid) const;
};

template<MemSpace M>
CELER_FUNCTION void RayleighLauncher<M>::operator()(ThreadId tid) const
{
    // Get views to Particle, and Physics
    ParticleTrackView particle(
        model.params.particle, model.states.particle, tid);

    MaterialTrackView material(
        model.params.material, model.states.material, tid);

    PhysicsTrackView physics(model.params.physics,
                             model.states.physics,
                             particle.particle_id(),
                             material.material_id(),
                             tid);

    // This interaction only applies if the Rayleigh model was selected
    if (physics.model_id() != rayleigh.model_id)
        return;

    RngEngine rng(model.states.rng, tid);

    // Assume only a single element in the material, for now
    CELER_ASSERT(material.material_view().num_elements() == 1);
    ElementId el_id{0};

    // Do the interaction
    RayleighInteractor interact(
        rayleigh, particle, model.states.direction[tid], el_id);

    model.states.interactions[tid] = interact(rng);
    CELER_ENSURE(model.states.interactions[tid]);
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
