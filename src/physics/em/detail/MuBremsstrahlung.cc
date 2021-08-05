//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MuBremsstrahlung.cc
//---------------------------------------------------------------------------//
#include "MuBremsstrahlung.hh"

#include "base/Assert.hh"
#include "base/KernelParamCalculator.cuda.hh"
#include "random/RngEngine.hh"
#include "physics/base/ModelInterface.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/PhysicsTrackView.hh"
#include "base/StackAllocator.hh"
#include "physics/material/MaterialTrackView.hh"
#include "MuBremsstrahlungInteractor.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
/*!
 * Interact using the Muon Bremsstrahlung model on applicable tracks.
 */
void mu_bremsstrahlung_interact(const MuBremsstrahlungInteractorPointers& mb,
                                const ModelInteractRefs<MemSpace::host>& model)
{
    for (auto tid : range(ThreadId{model.states.size()}))
    {
        StackAllocator<Secondary> allocate_secondaries(
            model.states.secondaries);
        ParticleTrackView particle(
            model.params.particle, model.states.particle, tid);

        // Setup for MaterialView access
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

        // This interaction only applies if the Muon Bremsstrahlung model was
        // selected
        if (physics.model_id() != mb.model_id)
            continue;

        // TODO: sample an element. For now assume one element per material
        const ElementComponentId   elcomp_id{0};
        MuBremsstrahlungInteractor interact(mb,
                                            particle,
                                            model.states.direction[tid],
                                            allocate_secondaries,
                                            material_view,
                                            elcomp_id);

        RngEngine rng(model.states.rng, tid);
        model.states.interactions[tid] = interact(rng);
        CELER_ENSURE(model.states.interactions[tid]);
    }
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
