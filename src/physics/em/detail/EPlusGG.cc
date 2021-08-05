//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EPlusGG.cc
//---------------------------------------------------------------------------//
#include "EPlusGG.hh"

#include "base/Assert.hh"
#include "base/KernelParamCalculator.cuda.hh"
#include "random/RngEngine.hh"
#include "physics/base/ModelInterface.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/PhysicsTrackView.hh"
#include "base/StackAllocator.hh"
#include "EPlusGGInteractor.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
/*!
 * Interact using the EPlusGG model on applicable tracks.
 */
void eplusgg_interact(const EPlusGGPointers&                   epgg,
                      const ModelInteractRefs<MemSpace::host>& model)
{
    for (auto tid : range(ThreadId{model.states.size()}))
    {
        // Get views to this Secondary, Particle, and Physics
        StackAllocator<Secondary> allocate_secondaries(
            model.states.secondaries);
        ParticleTrackView particle(
            model.params.particle, model.states.particle, tid);
        PhysicsTrackView physics(model.params.physics,
                                 model.states.physics,
                                 particle.particle_id(),
                                 MaterialId{},
                                 tid);

        // This interaction only applies if the EPlusGG model was selected
        if (physics.model_id() != epgg.model_id)
            continue;

        // Do the interaction
        EPlusGGInteractor interact(
            epgg, particle, model.states.direction[tid], allocate_secondaries);
        RngEngine rng(model.states.rng, tid);
        model.states.interactions[tid] = interact(rng);

        CELER_ENSURE(model.states.interactions[tid]);
    }
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
