//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CombinedBremLauncher.hh
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

#include "CombinedBremInteractor.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Model interactor kernel launcher
 */
template<MemSpace M>
struct CombinedBremLauncher
{
    CELER_FUNCTION
    CombinedBremLauncher(const CombinedBremNativeRef& data,
                         const ModelInteractRef<M>&   interaction)
        : shared(data), model(interaction)
    {
    }

    const CombinedBremNativeRef& shared; //!< Shared data for interactor
    const ModelInteractRef<M>&   model;  //!< State data needed to interact

    //! Create track views and launch interactor
    inline CELER_FUNCTION void operator()(ThreadId tid) const;
};

template<MemSpace M>
CELER_FUNCTION void CombinedBremLauncher<M>::operator()(ThreadId tid) const
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

    // This interaction only applies if the RelativisticBrem model was
    // selected
    if (physics.model_id() != shared.rb_data.ids.model)
        return;

    // Assume only a single element in the material, for now
    MaterialView material_view = material.material_view();
    CELER_ASSERT(material_view.num_elements() == 1);
    const ElementComponentId selected_element{0};

    CutoffView cutoffs(model.params.cutoffs, material.material_id());
    StackAllocator<Secondary> allocate_secondaries(model.states.secondaries);
    CombinedBremInteractor    interact(shared,
                                    model.params.lpm,
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
