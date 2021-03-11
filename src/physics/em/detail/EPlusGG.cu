//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EPlusGG.cu
//---------------------------------------------------------------------------//
#include "EPlusGG.hh"

#include "base/Assert.hh"
#include "base/KernelParamCalculator.cuda.hh"
#include "random/cuda/RngEngine.hh"
#include "physics/base/ModelInterface.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/PhysicsTrackView.hh"
#include "base/StackAllocator.hh"
#include "EPlusGGInteractor.hh"

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
 * Interact using the EPlusGG model on applicable tracks.
 */
__global__ void eplusgg_interact_kernel(const EPlusGGPointers       epgg,
                                        const ModelInteractPointers model)
{
    // Get the thread id
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= model.states.size())
        return;

    // Get views to this Secondary, Particle, and Physics
    StackAllocator<Secondary> allocate_secondaries(model.secondaries);
    ParticleTrackView      particle(
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
        epgg, particle, model.states.direction[tid.get()], allocate_secondaries);
    RngEngine rng(model.states.rng, tid);
    model.result[tid.get()] = interact(rng);

    CELER_ENSURE(model.result[tid.get()]);
}

} // namespace

//---------------------------------------------------------------------------//
// LAUNCHERS
//---------------------------------------------------------------------------//
/*!
 * Launch the EPlusGG interaction.
 */
void eplusgg_interact(const EPlusGGPointers&       eplusgg,
                      const ModelInteractPointers& model)
{
    CELER_EXPECT(eplusgg);
    CELER_EXPECT(model);

    // Calculate kernel launch params
    static const KernelParamCalculator calc_kernel_params(
        eplusgg_interact_kernel, "eplusgg_interact");
    auto params = calc_kernel_params(model.states.size());

    // Launch the kernel
    eplusgg_interact_kernel<<<params.grid_size, params.block_size>>>(eplusgg,
                                                                     model);
    CELER_CUDA_CHECK_ERROR();
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas