//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LivermorePE.cu
//---------------------------------------------------------------------------//
#include "LivermorePE.hh"

#include "base/KernelParamCalculator.cuda.hh"
#include "random/RngEngine.hh"
#include "physics/base/ModelInterface.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/PhysicsTrackView.hh"
#include "base/StackAllocator.hh"
#include "physics/material/ElementSelector.hh"
#include "physics/material/MaterialTrackView.hh"
#include "sim/SimTrackView.hh"
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
livermore_pe_interact_kernel(const LivermorePEDeviceRef                pe,
                             const RelaxationScratchDeviceRef&         scratch,
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
    CutoffView       cutoffs(model.params.cutoffs, material.material_id());
    SimTrackView     sim(model.states.sim, tid);

    // This interaction only applies if the Livermore PE model was selected
    if (physics.model_id() != pe.ids.model || !sim.alive())
        return;

    RngEngine rng(model.states.rng, tid);

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
                                   cutoffs,
                                   model.states.direction[tid],
                                   allocate_secondaries);

    model.states.interactions[tid] = interact(rng);
    CELER_ENSURE(model.states.interactions[tid]);
}

} // namespace

//---------------------------------------------------------------------------//
// LAUNCHERS
//---------------------------------------------------------------------------//
/*!
 * Launch the Livermore photoelectric interaction.
 */
void livermore_pe_interact(const LivermorePEDeviceRef&                pe,
                           const RelaxationScratchDeviceRef&          scratch,
                           const ModelInteractRefs<MemSpace::device>& model)
{
    CELER_EXPECT(pe);
    CELER_EXPECT(model);

    static const KernelParamCalculator calc_kernel_params(
        livermore_pe_interact_kernel, "livermore_pe_interact");
    auto params = calc_kernel_params(model.states.size());
    livermore_pe_interact_kernel<<<params.grid_size, params.block_size>>>(
        pe, scratch, model);
    CELER_CUDA_CHECK_ERROR();
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
