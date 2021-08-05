//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KleinNishina.cc
//---------------------------------------------------------------------------//
#include "KleinNishina.hh"

#include "base/Assert.hh"
#include "base/KernelParamCalculator.cuda.hh"
#include "random/RngEngine.hh"
#include "physics/base/ModelInterface.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/PhysicsTrackView.hh"
#include "base/StackAllocator.hh"
#include "KleinNishinaInteractor.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
/*!
 * Interact using the Klein-Nishina model on applicable tracks.
 */
void klein_nishina_interact(const KleinNishinaPointers&              kn,
                            const ModelInteractRefs<MemSpace::host>& model)
{
    for (auto tid : range(ThreadId{model.states.size()}))
    {
        StackAllocator<Secondary> allocate_secondaries(
            model.states.secondaries);
        ParticleTrackView particle(
            model.params.particle, model.states.particle, tid);

        PhysicsTrackView physics(model.params.physics,
                                 model.states.physics,
                                 particle.particle_id(),
                                 MaterialId{},
                                 tid);

        // This interaction only applies if the KN model was selected
        if (physics.model_id() != kn.model_id)
            continue;

        KleinNishinaInteractor interact(
            kn, particle, model.states.direction[tid], allocate_secondaries);

        RngEngine rng(model.states.rng, tid);
        model.states.interactions[tid] = interact(rng);
        CELER_ENSURE(model.states.interactions[tid]);
    }
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
