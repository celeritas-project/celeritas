//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file BetheHeitler.cc
//---------------------------------------------------------------------------//
#include "BetheHeitler.hh"

#include "base/Assert.hh"
#include "random/RngEngine.hh"
#include "physics/base/ModelInterface.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/PhysicsTrackView.hh"
#include "base/StackAllocator.hh"
#include "physics/material/MaterialTrackView.hh"
#include "BetheHeitlerInteractor.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
/*!
 * Interact using the Bethe-Heitler model on applicable tracks.
 */
void bethe_heitler_interact(const BetheHeitlerPointers&              bh,
                            const ModelInteractRefs<MemSpace::host>& model)
{
    for (auto tid : range(ThreadId{model.states.size()}))
    {
        StackAllocator<Secondary> allocate_secondaries(
            model.states.secondaries);
        ParticleTrackView particle(
            model.params.particle, model.states.particle, tid);

        // Setup for ElementView access
        MaterialTrackView material(
            model.params.material, model.states.material, tid);
        // Cache the associated MaterialView as function calls to
        // MaterialTrackView are expensive
        MaterialView material_view = material.material_view();

        PhysicsTrackView physics(model.params.physics,
                                 model.states.physics,
                                 particle.particle_id(),
                                 material.material_id(),
                                 tid);

        // This interaction only applies if the Bethe-Heitler model was
        // selected
        if (physics.model_id() != bh.model_id)
            continue;

        // Assume only a single element in the material, for now
        CELER_ASSERT(material_view.num_elements() == 1);
        ElementView element
            = material_view.element_view(celeritas::ElementComponentId{0});
        BetheHeitlerInteractor interact(bh,
                                        particle,
                                        model.states.direction[tid],
                                        allocate_secondaries,
                                        element);

        RngEngine rng(model.states.rng, tid);
        model.states.interactions[tid] = interact(rng);
        CELER_ENSURE(model.states.interactions[tid]);
    }
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
