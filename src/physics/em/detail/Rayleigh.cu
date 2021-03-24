//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Rayleigh.cu
//---------------------------------------------------------------------------//
#include "Rayleigh.hh"

#include "base/Assert.hh"
#include "base/KernelParamCalculator.cuda.hh"
#include "random/RngEngine.hh"
#include "physics/base/ModelInterface.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/material/MaterialTrackView.hh"
#include "physics/material/ElementView.hh"
#include "physics/material/MaterialView.hh"
#include "physics/base/PhysicsTrackView.hh"
#include "base/StackAllocator.hh"
#include "RayleighInteractor.hh"

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
 * Interact using the Rayleigh model on applicable tracks.
 */
__global__ void rayleigh_interact_kernel(const RayleighDeviceRef     rayleigh,
                                         const ModelInteractPointers model)
{
    // Get the thread id
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= model.states.size())
        return;

    // Get views to this Secondary, Particle, and Physics
    StackAllocator<Secondary> allocate_secondaries(model.secondaries);
    ParticleTrackView         particle(
        model.params.particle, model.states.particle, tid);

    MaterialTrackView material(
        model.params.material, model.states.material, tid);

    PhysicsTrackView physics(model.params.physics,
                             model.states.physics,
                             particle.particle_id(),
                             material.material_id(),
                             tid);

    // This interaction only applies if the Rayleigh model was selected
    if (physics.model_id() != rayleigh.model_id)
        return;

    RngEngine rng(model.states.rng, tid);

    MaterialView material_view = material.material_view();

    // Do the interaction
    RayleighInteractor interact(
        rayleigh,
        particle,
        model.states.direction[tid.get()],
        material_view.element_view(celeritas::ElementComponentId{0}));

    model.result[tid.get()] = interact(rng);
    CELER_ENSURE(model.result[tid.get()]);
}

} // namespace

//---------------------------------------------------------------------------//
// LAUNCHERS
//---------------------------------------------------------------------------//
/*!
 * Launch the Rayleigh interaction.
 */
void rayleigh_interact(const RayleighDeviceRef&     rayleigh,
                       const ModelInteractPointers& model)
{
    CELER_EXPECT(rayleigh);
    CELER_EXPECT(model);

    // Calculate kernel launch params
    static const KernelParamCalculator calc_kernel_params(
        rayleigh_interact_kernel, "rayleigh_interact");
    auto params = calc_kernel_params(model.states.size());

    // Launch the kernel
    rayleigh_interact_kernel<<<params.grid_size, params.block_size>>>(rayleigh,
                                                                      model);
    CELER_CUDA_CHECK_ERROR();
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
