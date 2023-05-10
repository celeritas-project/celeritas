//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/InitTracksLauncher.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/cont/Span.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/Types.hh"
#include "celeritas/geo/GeoMaterialView.hh"
#include "celeritas/geo/GeoTrackView.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/mat/MaterialTrackView.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/phys/PhysicsTrackView.hh"
#include "celeritas/track/TrackInitData.hh"

#include "../SimTrackView.hh"
#include "Utils.hh"

#if !CELER_DEVICE_COMPILE
#    include "corecel/cont/ArrayIO.hh"
#endif

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Initialize the track states.
 *
 * The track initializers are created from either primary particles or
 * secondaries. The new tracks are inserted into empty slots (vacancies) in the
 * track vector.
 */
template<MemSpace M>
class InitTracksLauncher
{
  public:
    //!@{
    //! \name Type aliases
    using ParamsRef = CoreParamsData<Ownership::const_reference, M>;
    using StateRef = CoreStateData<Ownership::reference, M>;
    //!@}

  public:
    // Construct with shared and state data
    CELER_FUNCTION InitTracksLauncher(ParamsRef const& params,
                                      StateRef const& states,
                                      size_type /* num_vacancies */)
        : params_(params), states_(states)
    {
        CELER_EXPECT(params_);
        CELER_EXPECT(states_);
    }

    // Initialize track states
    inline CELER_FUNCTION void operator()(ThreadId tid) const;

  private:
    ParamsRef const& params_;
    StateRef const& states_;
};

//---------------------------------------------------------------------------//
/*!
 * Initialize the track states.
 */
template<MemSpace M>
CELER_FUNCTION void InitTracksLauncher<M>::operator()(ThreadId tid) const
{
    // Get the track initializer from the back of the vector. Since new
    // initializers are pushed to the back of the vector, these will be the
    // most recently added and therefore the ones that still might have a
    // parent they can copy the geometry state from.
    auto const& data = states_.init;
    ItemId<TrackInitializer> idx{
        index_before(data.scalars.num_initializers, tid)};
    TrackInitializer const& init = data.initializers[idx];

    // Thread ID of vacant track where the new track will be initialized
    TrackSlotId vacancy = [&] {
        TrackSlotId idx{index_before(data.scalars.num_vacancies, tid)};
        return data.vacancies[idx];
    }();

    // Initialize the simulation state
    {
        SimTrackView sim(params_.sim, states_.sim, vacancy);
        sim = init.sim;
    }

    // Initialize the particle physics data
    {
        ParticleTrackView particle(
            params_.particles, states_.particles, vacancy);
        particle = init.particle;
    }

    // Initialize the geometry
    {
        GeoTrackView geo(params_.geometry, states_.geometry, vacancy);
        if (tid < data.scalars.num_secondaries)
        {
            // Copy the geometry state from the parent for improved
            // performance
            TrackSlotId parent_id = data.parents[TrackSlotId{
                index_before(data.parents.size(), tid)}];
            GeoTrackView parent(params_.geometry, states_.geometry, parent_id);
            geo = GeoTrackView::DetailedInitializer{parent, init.geo.dir};
        }
        else
        {
            // Initialize it from the position (more expensive)
            geo = init.geo;
#if !CELER_DEVICE_COMPILE
            // TODO: kill particle with 'error' state if this happens
            CELER_VALIDATE(!geo.is_outside(),
                           << "track started outside the geometry at "
                           << init.geo.pos);
#endif
        }

        // Initialize the material
        GeoMaterialView geo_mat(params_.geo_mats);
        MaterialTrackView mat(params_.materials, states_.materials, vacancy);
        mat = {geo_mat.material_id(geo.volume_id())};
    }

    // Initialize the physics state
    {
        PhysicsTrackView phys(
            params_.physics, states_.physics, {}, {}, vacancy);
        phys = {};
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
