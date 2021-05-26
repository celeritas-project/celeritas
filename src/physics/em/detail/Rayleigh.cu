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
#include "physics/material/Types.hh"
#include "physics/material/MaterialTrackView.hh"
#include "physics/material/ElementView.hh"
#include "physics/material/ElementSelector.hh"
#include "physics/base/PhysicsTrackView.hh"
#include "sim/SimTrackView.hh"
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
__global__ void
rayleigh_interact_kernel(const RayleighDeviceRef                   rayleigh,
                         const ModelInteractRefs<MemSpace::device> model)
{
    // Get the thread id
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (!(tid < model.states.size()))
        return;

    // Get views to Particle, and Physics
    ParticleTrackView particle(
        model.params.particle, model.states.particle, tid);

    MaterialTrackView material(
        model.params.material, model.states.material, tid);

    PhysicsTrackView physics(model.params.physics,
                             model.states.physics,
                             particle.particle_id(),
                             material.material_id(),
                             tid);
    SimTrackView     sim(model.states.sim, tid);

    // This interaction only applies if the Rayleigh model was selected
    if (physics.model_id() != rayleigh.model_id || !sim.alive())
        return;

    RngEngine rng(model.states.rng, tid);

    // Assume only a single element in the material, for now
    CELER_ASSERT(material.material_view().num_elements() == 1);
    ElementId el_id{0};

    // Do the interaction
    RayleighInteractor interact(
        rayleigh, particle, model.states.direction[tid], el_id);

    model.states.interactions[tid] = interact(rng);
    CELER_ENSURE(model.states.interactions[tid]);
}

} // namespace

//---------------------------------------------------------------------------//
// LAUNCHERS
//---------------------------------------------------------------------------//
/*!
 * Launch the Rayleigh interaction.
 */
void rayleigh_interact(const RayleighDeviceRef&                   rayleigh,
                       const ModelInteractRefs<MemSpace::device>& model)
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
