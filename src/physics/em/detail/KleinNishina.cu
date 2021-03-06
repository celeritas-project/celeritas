//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KleinNishina.cu
//---------------------------------------------------------------------------//
#include "KleinNishina.hh"

#include "base/Assert.hh"
#include "base/KernelParamCalculator.cuda.hh"
#include "random/cuda/RngEngine.hh"
#include "physics/base/ModelInterface.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/PhysicsTrackView.hh"
#include "base/StackAllocator.hh"
#include "KleinNishinaInteractor.hh"

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
 * Interact using the Klein-Nishina model on applicable tracks.
 */
__global__ void klein_nishina_interact_kernel(const KleinNishinaPointers  kn,
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

    // This interaction only applies if the KN model was selected
    if (physics.model_id() != kn.model_id)
        return;

    KleinNishinaInteractor interact(
        kn, particle, ptrs.states.direction[tid.get()], allocate_secondaries);

    RngEngine rng(ptrs.states.rng, tid);
    ptrs.result[tid.get()] = interact(rng);
    CELER_ENSURE(ptrs.result[tid.get()]);
}

} // namespace

//---------------------------------------------------------------------------//
// LAUNCHERS
//---------------------------------------------------------------------------//
/*!
 * Launch the KN interaction.
 */
void klein_nishina_interact(const KleinNishinaPointers&  kn,
                            const ModelInteractPointers& model)
{
    CELER_EXPECT(kn);
    CELER_EXPECT(model);

    static const KernelParamCalculator calc_kernel_params(
        klein_nishina_interact_kernel, "klein_nishina_interact");
    auto                  params = calc_kernel_params(model.states.size());
    klein_nishina_interact_kernel<<<params.grid_size, params.block_size>>>(
        kn, model);
    CELER_CUDA_CHECK_ERROR();
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
