//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LivermorePE.cc
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
#include "LivermorePEInteractor.hh"
#include "LivermorePEMicroXsCalculator.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
/*!
 * Interact using the Livermore photoelectric model on applicable tracks.
 */
void livermore_pe_interact(const LivermorePEHostRef&                pe,
                           const RelaxationScratchHostRef&          scratch,
                           const ModelInteractRefs<MemSpace::host>& model)
{
    for (auto tid : range(ThreadId{model.states.size()}))
    {
        StackAllocator<Secondary> allocate_secondaries(
            model.states.secondaries);
        ParticleTrackView particle(
            model.params.particle, model.states.particle, tid);
        MaterialTrackView material(
            model.params.material, model.states.material, tid);
        PhysicsTrackView physics(model.params.physics,
                                 model.states.physics,
                                 particle.particle_id(),
                                 material.material_id(),
                                 tid);
        CutoffView       cutoffs(model.params.cutoffs, material.material_id());

        // This interaction only applies if the Livermore PE model was selected
        if (physics.model_id() != pe.ids.model)
            continue;

        RngEngine rng(model.states.rng, tid);

        // Sample an element
        ElementSelector select_el(
            material.material_view(),
            LivermorePEMicroXsCalculator{pe, particle.energy()},
            material.element_scratch());
        ElementComponentId comp_id = select_el(rng);
        ElementId el_id = material.material_view().element_id(comp_id);

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
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
