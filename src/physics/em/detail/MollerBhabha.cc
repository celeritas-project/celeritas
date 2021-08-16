//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MollerBhabha.cc
//---------------------------------------------------------------------------//
#include "MollerBhabha.hh"

#include "base/Assert.hh"
#include "base/StackAllocator.hh"
#include "physics/base/CutoffView.hh"
#include "physics/base/ModelInterface.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/PhysicsTrackView.hh"
#include "physics/material/MaterialTrackView.hh"
#include "random/RngEngine.hh"
#include "MollerBhabhaInteractor.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
/*!
 * Interact using the Moller-Bhabha model on applicable tracks.
 */
void moller_bhabha_interact(const MollerBhabhaPointers&              mb,
                            const ModelInteractRefs<MemSpace::host>& model)
{
    for (auto tid : range(ThreadId{model.states.size()}))
    {
        StackAllocator<Secondary> allocate_secondaries(
            model.states.secondaries);
        ParticleTrackView particle(
            model.params.particle, model.states.particle, tid);

        MaterialTrackView material(
            model.params.material, model.states.material, tid);

        PhysicsTrackView physics(model.params.physics,
                                 model.states.physics,
                                 particle.particle_id(),
                                 material.material_id(),
                                 tid);

        CutoffView cutoff(model.params.cutoffs, material.material_id());

        // This interaction only applies if the MB model was selected
        if (physics.model_id() != mb.model_id)
        {
            continue;
        }

        MollerBhabhaInteractor interact(mb,
                                        particle,
                                        cutoff,
                                        model.states.direction[tid],
                                        allocate_secondaries);

        RngEngine rng(model.states.rng, tid);
        model.states.interactions[tid] = interact(rng);
        CELER_ENSURE(model.states.interactions[tid]);
    }
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
