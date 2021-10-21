//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LivermorePELauncher.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Assert.hh"
#include "base/Macros.hh"
#include "base/StackAllocator.hh"
#include "base/Types.hh"
#include "physics/base/ModelData.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/PhysicsTrackView.hh"
#include "physics/material/ElementSelector.hh"
#include "physics/material/MaterialTrackView.hh"
#include "random/RngEngine.hh"
#include "LivermorePEInteractor.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Model interactor kernel launcher
 */
template<MemSpace M>
struct LivermorePELauncher
{
    CELER_FUNCTION LivermorePELauncher(const LivermorePEData&      pointers,
                                       const ModelInteractRefs<M>& interaction)
        : pe(pointers), model(interaction)
    {
    }

    const LivermorePEData&      pe;    //!< Shared data for interactor
    const ModelInteractRefs<M>& model; //!< State data needed to interact

    //! Create track views and launch interactor
    inline CELER_FUNCTION void operator()(ThreadId tid) const;
};

template<MemSpace M>
CELER_FUNCTION void LivermorePELauncher<M>::operator()(ThreadId tid) const
{
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

    // This interaction only applies if the Livermore PE model was selected
    if (physics.model_id() != pe.ids.model)
        return;

    RngEngine rng(model.states.rng, tid);

    // Sample an element
    ElementSelector select_el(
        material.material_view(),
        LivermorePEMicroXsCalculator{pe, particle.energy()},
        material.element_scratch());
    ElementComponentId comp_id = select_el(rng);
    ElementId          el_id   = material.material_view().element_id(comp_id);

    AtomicRelaxationHelper relaxation(
        model.params.relaxation, model.states.relaxation, el_id, tid);
    LivermorePEInteractor interact(pe,
                                   relaxation,
                                   el_id,
                                   particle,
                                   cutoffs,
                                   model.states.direction[tid],
                                   allocate_secondaries);

    model.states.interactions[tid] = interact(rng);
    CELER_ENSURE(model.states.interactions[tid]);
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
