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
#include "random/RngEngine.hh"
#include "physics/base/ModelInterface.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/PhysicsTrackView.hh"
#include "physics/material/MaterialTrackView.hh"
#include "physics/base/CutoffView.hh"
#include "base/StackAllocator.hh"
#include "sim/SimTrackView.hh"
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
__global__ void
moller_bhabha_interact_kernel(const MollerBhabhaPointers                mb,
                              const ModelInteractRefs<MemSpace::device> model)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (!(tid < model.states.size()))
        return;

    StackAllocator<Secondary> allocate_secondaries(model.states.secondaries);
    ParticleTrackView         particle(
        model.params.particle, model.states.particle, tid);

    MaterialTrackView material(
        model.params.material, model.states.material, tid);

    PhysicsTrackView physics(model.params.physics,
                             model.states.physics,
                             particle.particle_id(),
                             material.material_id(),
                             tid);

    CutoffView   cutoff(model.params.cutoffs, material.material_id());
    SimTrackView sim(model.states.sim, tid);

    // This interaction only applies if the MB model was selected
    if (physics.model_id() != mb.model_id || !sim.alive())
    {
        return;
    }

    MollerBhabhaInteractor interact(
        mb, particle, cutoff, model.states.direction[tid], allocate_secondaries);

    RngEngine rng(model.states.rng, tid);
    model.states.interactions[tid] = interact(rng);
    CELER_ENSURE(model.states.interactions[tid]);
}

} // namespace

//---------------------------------------------------------------------------//
// LAUNCHERS
//---------------------------------------------------------------------------//
/*!
 * Launch the MB interaction.
 */
void moller_bhabha_interact(const MollerBhabhaPointers&                mb,
                            const ModelInteractRefs<MemSpace::device>& model)
{
    CELER_EXPECT(mb);
    CELER_EXPECT(model);

    static const KernelParamCalculator calc_kernel_params(
        moller_bhabha_interact_kernel, "moller_bhabha_interact");
    auto params = calc_kernel_params(model.states.size());
    moller_bhabha_interact_kernel<<<params.grid_size, params.block_size>>>(
        mb, model);
    CELER_CUDA_CHECK_ERROR();
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
