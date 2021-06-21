//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file BetheBloch.cu
//---------------------------------------------------------------------------//
#include "BetheBloch.hh"

#include "base/Assert.hh"
#include "base/KernelParamCalculator.cuda.hh"
#include "random/RngEngine.hh"
#include "physics/base/ModelInterface.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/PhysicsTrackView.hh"
#include "base/StackAllocator.hh"
#include "physics/material/MaterialTrackView.hh"
#include "BetheBlochInteractor.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
/*!
 * Interact using the Bethe-Bloch model on applicable tracks.
 */
__global__ void bethe_bloch_interact_kernel(const BetheBlochInteractorPointers  bb,
                                              const ModelInteractRefs<MemSpace::device> model)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= model.states.size())
        return;

    StackAllocator<Secondary> allocate_secondaries(model.states.secondaries);
    ParticleTrackView particle(model.params.particle, model.states.particle, tid);

    // Setup for MaterialView access
    MaterialTrackView material(model.params.material, model.states.material, tid);
    // Cache the associated MaterialView as function calls to MaterialTrackView
    // are expensive
    MaterialView material_view = material.material_view();

    PhysicsTrackView physics(model.params.physics,
                             model.states.physics,
                             particle.particle_id(),
                             material.material_id(),
                             tid);

    // This interaction only applies if the Bethe-Bloch model was selected
    if (physics.model_id() != bb.model_id)
        return;

    BetheBlochInteractor interact(
        bb,
        particle,
        model.states.direction[tid],
        allocate_secondaries,
        material_view);

    RngEngine rng(model.states.rng, tid);
    model.states.interactions[tid] = interact(rng);
    CELER_ENSURE(model.states.interactions[tid]);
}

} // namespace

//---------------------------------------------------------------------------//
// LAUNCHERS
//---------------------------------------------------------------------------//
/*!
 * Launch the Bethe-Bloch interaction.
 */
void bethe_bloch_interact(const BetheBlochInteractorPointers&  bb,
                          const ModelInteractRefs<MemSpace::device>& model)
{
    CELER_EXPECT(bb);
    CELER_EXPECT(model);

    static const KernelParamCalculator calc_kernel_params(
        bethe_bloch_interact_kernel, "bethe_bloch_interact");
    auto params = calc_kernel_params(model.states.size());
    bethe_bloch_interact_kernel<<<params.grid_size, params.block_size>>>(
        bb, model);
    CELER_CUDA_CHECK_ERROR();
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

