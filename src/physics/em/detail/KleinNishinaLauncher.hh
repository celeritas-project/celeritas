//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KleinNishinaLauncher.hh
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
#include "random/RngEngine.hh"

#include "KleinNishinaInteractor.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Model interactor kernel launcher
 */
template<MemSpace M>
struct KleinNishinaLauncher
{
    CELER_FUNCTION KleinNishinaLauncher(const KleinNishinaData&    data,
                                        const ModelInteractRef<M>& interaction)
        : kn(data), model(interaction)
    {
    }

    const KleinNishinaData&    kn;    //!< Shared data for interactor
    const ModelInteractRef<M>& model; //!< State data needed to interact

    //! Create track views and launch interactor
    inline CELER_FUNCTION void operator()(ThreadId tid) const;
};

template<MemSpace M>
CELER_FUNCTION void KleinNishinaLauncher<M>::operator()(ThreadId tid) const
{
    StackAllocator<Secondary> allocate_secondaries(model.states.secondaries);
    ParticleTrackView         particle(
        model.params.particle, model.states.particle, tid);

    PhysicsTrackView physics(model.params.physics,
                             model.states.physics,
                             particle.particle_id(),
                             MaterialId{},
                             tid);

    // This interaction only applies if the KN model was selected
    if (physics.model_id() != kn.model_id)
        return;

    KleinNishinaInteractor interact(
        kn, particle, model.states.direction[tid], allocate_secondaries);

    RngEngine rng(model.states.rng, tid);
    model.states.interactions[tid] = interact(rng);
    CELER_ENSURE(model.states.interactions[tid]);
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
