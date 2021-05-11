//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SeltzerBerger.cu
//---------------------------------------------------------------------------//
#include "SeltzerBerger.hh"

#include "base/Assert.hh"
#include "base/KernelParamCalculator.cuda.hh"
#include "random/RngEngine.hh"
#include "physics/base/ModelInterface.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/PhysicsTrackView.hh"
#include "base/StackAllocator.hh"
#include "physics/material/MaterialTrackView.hh"
#include "SeltzerBergerInteractor.hh"

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
 * Interact using the Seltzer-Berger model on applicable tracks.
 */
__global__ void seltzer_berger_interact_kernel(
    const SeltzerBergerDeviceRef&              device_pointers,
    const ModelInteractRefs<MemSpace::device>& interaction)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (!(tid < interaction.states.size()))
        return;

    StackAllocator<Secondary> allocate_secondaries(
        interaction.states.secondaries);
    ParticleTrackView particle(
        interaction.params.particle, interaction.states.particle, tid);

    // Setup for ElementView access
    MaterialTrackView material(
        interaction.params.material, interaction.states.material, tid);
    // Cache the associated MaterialView as function calls to MaterialTrackView
    // are expensive
    MaterialView material_view = material.material_view();

    PhysicsTrackView physics(interaction.params.physics,
                             interaction.states.physics,
                             particle.particle_id(),
                             material.material_id(),
                             tid);

    // This interaction only applies if the Seltzer-Berger model was selected
    if (physics.model_id() != device_pointers.ids.model)
        return;

    CutoffView cutoffs(interaction.params.cutoffs, material.material_id());

    // Assume only a single element in the material, for now
    CELER_ASSERT(material_view.num_elements() == 1);
    SeltzerBergerInteractor interact(device_pointers,
                                     particle,
                                     interaction.states.direction[tid],
                                     cutoffs,
                                     allocate_secondaries,
                                     material_view);

    RngEngine rng(interaction.states.rng, tid);
    interaction.states.interactions[tid] = interact(rng);
    CELER_ENSURE(interaction.states.interactions[tid]);
}

} // namespace

//---------------------------------------------------------------------------//
// LAUNCHERS
//---------------------------------------------------------------------------//
/*!
 * Launch the Seltzer-Berger interaction.
 */
void seltzer_berger_interact(
    const SeltzerBergerDeviceRef&              device_pointers,
    const ModelInteractRefs<MemSpace::device>& interaction)
{
    CELER_EXPECT(device_pointers);
    CELER_EXPECT(interaction);

    static const KernelParamCalculator calc_kernel_params(
        seltzer_berger_interact_kernel, "seltzer_berger_interact");
    auto params = calc_kernel_params(interaction.states.size());
    seltzer_berger_interact_kernel<<<params.grid_size, params.block_size>>>(
        device_pointers, interaction);
    CELER_CUDA_CHECK_ERROR();
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
