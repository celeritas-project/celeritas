//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MuBremsstrahlung.cu
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
namespace
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
/*!
 * Interact using the Muon Bremsstrahlung model on applicable tracks.
 */
__global__ void mu_bremsstrahlung_interact_kernel(
    const MuBremsstrahlungInteractorPointers  mb,
    const ModelInteractRefs<MemSpace::device> model)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= model.states.size())
        return;

    StackAllocator<Secondary> allocate_secondaries(model.states.secondaries);
    ParticleTrackView         particle(
        model.params.particle, model.states.particle, tid);

    // Setup for MaterialView access
    MaterialTrackView material(
        model.params.material, model.states.material, tid);
    // Cache the associated MaterialView as function calls to MaterialTrackView
    // are expensive
    MaterialView material_view = material.material_view();

    PhysicsTrackView physics(model.params.physics,
                             model.states.physics,
                             particle.particle_id(),
                             material.material_id(),
                             tid);

    // This interaction only applies if the Muon Bremsstrahlung model was
    // selected
    if (physics.model_id() != mb.model_id)
        return;

    ElementView element
        = material_view.element_view(celeritas::ElementComponentId{0});
    MuBremsstrahlungInteractor interact(mb,
                                        particle,
                                        model.states.direction[tid],
                                        allocate_secondaries,
                                        element);

    RngEngine rng(model.states.rng, tid);
    model.states.interactions[tid] = interact(rng);
    CELER_ENSURE(model.states.interactions[tid]);
}

} // namespace

//---------------------------------------------------------------------------//
// LAUNCHERS
//---------------------------------------------------------------------------//
/*!
 * Launch the Muon Bremsstrahlung interaction.
 */
void mu_bremsstrahlung_interact(const MuBremsstrahlungInteractorPointers& mb,
                                const ModelInteractRefs<MemSpace::device>& model)
{
    CELER_EXPECT(mb);
    CELER_EXPECT(model);

    static const KernelParamCalculator calc_kernel_params(
        mu_bremsstrahlung_interact_kernel, "mu_bremsstrahlung_interact");
    auto params = calc_kernel_params(model.states.size());
    mu_bremsstrahlung_interact_kernel<<<params.grid_size, params.block_size>>>(
        mb, model);
    CELER_CUDA_CHECK_ERROR();
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

