//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SeltzerBerger.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Assert.hh"
#include "base/Macros.hh"
#include "base/StackAllocator.hh"
#include "base/Types.hh"
#include "physics/base/ModelData.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/PhysicsTrackView.hh"
#include "physics/material/MaterialTrackView.hh"
#include "random/RngEngine.hh"
#include "SeltzerBergerInteractor.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Model interactor kernel launcher
 */
template<MemSpace M>
struct SeltzerBergerLauncher
{
    CELER_FUNCTION
    SeltzerBergerLauncher(const SeltzerBergerNativeRef& pointers,
                          const ModelInteractRefs<M>&   interaction)
        : sb(pointers), model(interaction)
    {
    }

    const SeltzerBergerNativeRef& sb;    //!< Shared data for interactor
    const ModelInteractRefs<M>&   model; //!< State data needed to interact

    //! Create track views and launch interactor
    inline CELER_FUNCTION void operator()(ThreadId tid) const;
};

template<MemSpace M>
CELER_FUNCTION void SeltzerBergerLauncher<M>::operator()(ThreadId tid) const
{
    ParticleTrackView particle(
        model.params.particle, model.states.particle, tid);

    // Setup for ElementView access
    MaterialTrackView material(
        model.params.material, model.states.material, tid);

    PhysicsTrackView physics(model.params.physics,
                             model.states.physics,
                             particle.particle_id(),
                             material.material_id(),
                             tid);

    // This interaction only applies if the Seltzer-Berger model was
    // selected
    if (physics.model_id() != sb.ids.model)
        return;

    // Assume only a single element in the material, for now
    MaterialView material_view = material.material_view();
    CELER_ASSERT(material_view.num_elements() == 1);
    const ElementComponentId selected_element{0};

    CutoffView cutoffs(model.params.cutoffs, material.material_id());
    StackAllocator<Secondary> allocate_secondaries(model.states.secondaries);
    SeltzerBergerInteractor   interact(sb,
                                     particle,
                                     model.states.direction[tid],
                                     cutoffs,
                                     allocate_secondaries,
                                     material_view,
                                     selected_element);

    RngEngine rng(model.states.rng, tid);
    model.states.interactions[tid] = interact(rng);
    CELER_ENSURE(model.states.interactions[tid]);
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
