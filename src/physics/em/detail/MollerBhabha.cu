//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MollerBhabha.cu
//---------------------------------------------------------------------------//
#include "MollerBhabha.hh"

#include "base/Assert.hh"
#include "base/KernelParamCalculator.cuda.hh"
#include "random/cuda/RngEngine.hh"
#include "physics/base/ModelInterface.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/PhysicsTrackView.hh"
#include "base/StackAllocator.hh"
#include "MollerBhabhaInteractor.hh"

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
 * Interact using the Moller-Bhabha model on applicable tracks.
 */
__global__ void moller_bhabha_interact_kernel(const MollerBhabhaPointers  mb,
                                              const ModelInteractPointers ptrs)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= ptrs.states.size())
        return;

    StackAllocator<Secondary> allocate_secondaries(ptrs.secondaries);
    ParticleTrackView particle(ptrs.params.particle, ptrs.states.particle, tid);

    PhysicsTrackView physics(ptrs.params.physics,
                             ptrs.states.physics,
                             particle.particle_id(),
                             MaterialId{},
                             tid);

    // This interaction only applies if the MB model was selected
    if (physics.model_id() != mb.model_id)
        return;

    MollerBhabhaInteractor interact(
        mb, particle, ptrs.states.direction[tid.get()], allocate_secondaries);

    RngEngine rng(ptrs.states.rng, tid);
    ptrs.result[tid.get()] = interact(rng);
    CELER_ENSURE(ptrs.result[tid.get()]);
}

} // namespace

//---------------------------------------------------------------------------//
// LAUNCHERS
//---------------------------------------------------------------------------//
/*!
 * Launch the MB interaction.
 */
void moller_bhabha_interact(const MollerBhabhaPointers&  mb,
                            const ModelInteractPointers& model)
{
    CELER_EXPECT(mb);
    CELER_EXPECT(model);

    static const KernelParamCalculator calc_kernel_params(
        moller_bhabha_interact_kernel, "moller_bhabha_interact");
    auto                  params = calc_kernel_params(model.states.size());
    moller_bhabha_interact_kernel<<<params.grid_size, params.block_size>>>(
        mb, model);
    CELER_CUDA_CHECK_ERROR();
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
