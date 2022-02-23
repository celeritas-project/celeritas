//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EPlusGGLauncher.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Assert.hh"
#include "base/StackAllocator.hh"
#include "base/Types.hh"
#include "random/RngEngine.hh"
#include "physics/base/ModelData.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/PhysicsTrackView.hh"
#include "EPlusGGInteractor.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Model interactor kernel launcher
 */
template<MemSpace M>
struct EPlusGGLauncher
{
    CELER_FUNCTION EPlusGGLauncher(const EPlusGGData&         data,
                                   const ModelInteractRef<M>& interaction)
        : epgg(data), model(interaction)
    {
    }

    const EPlusGGData&         epgg;  //!< Shared data for interactor
    const ModelInteractRef<M>& model; //!< State data needed to interact

    //! Create track views and launch interactor
    inline CELER_FUNCTION void operator()(ThreadId tid) const;
};

template<MemSpace M>
CELER_FUNCTION void EPlusGGLauncher<M>::operator()(ThreadId tid) const
{
    // Get views to this Secondary, Particle, and Physics
    StackAllocator<Secondary> allocate_secondaries(model.states.secondaries);
    ParticleTrackView         particle(
        model.params.particle, model.states.particle, tid);
    PhysicsTrackView physics(model.params.physics,
                             model.states.physics,
                             particle.particle_id(),
                             MaterialId{},
                             tid);

    // This interaction only applies if the EPlusGG model was selected
    if (physics.model_id() != epgg.model_id)
        return;

    // Do the interaction
    EPlusGGInteractor interact(
        epgg, particle, model.states.direction[tid], allocate_secondaries);
    RngEngine rng(model.states.rng, tid);
    model.states.interactions[tid] = interact(rng);

    CELER_ENSURE(model.states.interactions[tid]);
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
