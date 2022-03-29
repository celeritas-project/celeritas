//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file BetheHeitlerLauncher.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Assert.hh"
#include "base/Macros.hh"
#include "base/StackAllocator.hh"
#include "base/Types.hh"
#include "physics/base/ModelData.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/PhysicsTrackView.hh"
#include "physics/base/Types.hh"
#include "physics/material/MaterialTrackView.hh"
#include "random/RngEngine.hh"

#include "BetheHeitlerInteractor.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Model interactor kernel launcher
 */
template<MemSpace M>
struct BetheHeitlerLauncher
{
    CELER_FUNCTION BetheHeitlerLauncher(const BetheHeitlerData&    data,
                                        const ModelInteractRef<M>& interaction)
        : bh(data), model(interaction)
    {
    }

    const BetheHeitlerData&    bh;    //!< Shared data for interactor
    const ModelInteractRef<M>& model; //!< State data needed to interact

    //! Create track views and launch interactor
    inline CELER_FUNCTION void operator()(ThreadId tid) const;
};

template<MemSpace M>
CELER_FUNCTION void BetheHeitlerLauncher<M>::operator()(ThreadId tid) const
{
    StackAllocator<Secondary> allocate_secondaries(model.states.secondaries);
    ParticleTrackView         particle(
        model.params.particle, model.states.particle, tid);

    // Setup for ElementView access
    MaterialTrackView material(
        model.params.material, model.states.material, tid);

    PhysicsTrackView physics(model.params.physics,
                             model.states.physics,
                             particle.particle_id(),
                             material.material_id(),
                             tid);

    // This interaction only applies if the Bethe-Heitler model was
    // selected
    if (physics.model_id() != bh.model_id)
        return;

    // Cache the associated MaterialView as function calls to
    // MaterialTrackView are expensive
    MaterialView material_view = material.make_material_view();

    // Assume only a single element in the material, for now
    CELER_ASSERT(material_view.num_elements() == 1);
    ElementView element
        = material_view.make_element_view(celeritas::ElementComponentId{0});
    BetheHeitlerInteractor interact(bh,
                                    particle,
                                    model.states.direction[tid],
                                    allocate_secondaries,
                                    material_view,
                                    element);

    RngEngine rng(model.states.rng, tid);
    model.states.interactions[tid] = interact(rng);
    CELER_ENSURE(model.states.interactions[tid]);
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
