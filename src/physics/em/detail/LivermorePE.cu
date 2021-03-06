//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LivermorePE.cu
//---------------------------------------------------------------------------//
#include "LivermorePE.hh"

#include "base/KernelParamCalculator.cuda.hh"
#include "random/cuda/RngEngine.hh"
#include "physics/base/ModelInterface.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/PhysicsTrackView.hh"
#include "base/StackAllocator.hh"
#include "physics/material/ElementSelector.hh"
#include "physics/material/MaterialTrackView.hh"
#include "LivermorePEInteractor.hh"
#include "LivermorePEMicroXsCalculator.hh"

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
 * Interact using the Livermore photoelectric model on applicable tracks.
 */
__global__ void
livermore_pe_interact_kernel(const LivermorePEPointers        pe,
                             const RelaxationScratchPointers& scratch,
                             const ModelInteractPointers      ptrs)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= ptrs.states.size())
        return;

    StackAllocator<Secondary> allocate_secondaries(ptrs.secondaries);
    ParticleTrackView particle(ptrs.params.particle, ptrs.states.particle, tid);
    MaterialTrackView material(ptrs.params.material, ptrs.states.material, tid);
    PhysicsTrackView  physics(ptrs.params.physics,
                             ptrs.states.physics,
                             particle.particle_id(),
                             material.material_id(),
                             tid);

    // This interaction only applies if the Livermore PE model was selected
    if (physics.model_id() != pe.model_id)
        return;

    RngEngine rng(ptrs.states.rng, tid);

    // Sample an element
    ElementSelector select_el(
        material.material_view(),
        LivermorePEMicroXsCalculator{pe, particle.energy()},
        material.element_scratch());
    ElementComponentId comp_id = select_el(rng);
    ElementId          el_id   = material.material_view().element_id(comp_id);

    LivermorePEInteractor interact(pe,
                                   scratch,
                                   el_id,
                                   particle,
                                   ptrs.states.direction[tid.get()],
                                   allocate_secondaries);

    ptrs.result[tid.get()] = interact(rng);
    CELER_ENSURE(ptrs.result[tid.get()]);
}

} // namespace

//---------------------------------------------------------------------------//
// LAUNCHERS
//---------------------------------------------------------------------------//
/*!
 * Launch the Livermore photoelectric interaction.
 */
void livermore_pe_interact(const LivermorePEPointers&       pe,
                           const RelaxationScratchPointers& scratch,
                           const ModelInteractPointers&     model)
{
    CELER_EXPECT(pe);
    CELER_EXPECT(model);

    static const KernelParamCalculator calc_kernel_params(
        livermore_pe_interact_kernel, "livermore_pe_interact");
    auto                  params = calc_kernel_params(model.states.size());
    livermore_pe_interact_kernel<<<params.grid_size, params.block_size>>>(
        pe, scratch, model);
    CELER_CUDA_CHECK_ERROR();
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
