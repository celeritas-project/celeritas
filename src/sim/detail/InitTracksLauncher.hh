//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file InitTracksLauncher.hh
//---------------------------------------------------------------------------//
#pragma once

#include "geometry/GeoMaterialView.hh"
#include "geometry/GeoTrackView.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/PhysicsTrackView.hh"
#include "physics/material/MaterialTrackView.hh"
#include "sim/TrackData.hh"
#include "sim/TrackInitData.hh"
#include "sim/SimTrackView.hh"
#include "Utils.hh"

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
    //! Type aliases
    using ParamsDataRef         = ParamsData<Ownership::const_reference, M>;
    using StateDataRef          = StateData<Ownership::reference, M>;
    using TrackInitStateDataRef = TrackInitStateData<Ownership::reference, M>;
    //!@}

  public:
    // Construct with shared and state data
    CELER_FUNCTION InitTracksLauncher(const ParamsDataRef&         params,
                                      const StateDataRef&          states,
                                      const TrackInitStateDataRef& data)
        : params_(params), states_(states), data_(data)
    {
        CELER_EXPECT(params_);
        CELER_EXPECT(states_);
        CELER_EXPECT(data_);
    }

    // Initialize track states
    inline CELER_FUNCTION void operator()(ThreadId tid) const;

  private:
    const ParamsDataRef&         params_;
    const StateDataRef&          states_;
    const TrackInitStateDataRef& data_;
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
    const TrackInitializer& init
        = data_.initializers[from_back(data_.initializers.size(), tid)];

    // Thread ID of vacant track where the new track will be initialized
    ThreadId vac_id(data_.vacancies[from_back(data_.vacancies.size(), tid)]);

    // Initialize the simulation state
    {
        SimTrackView sim(states_.sim, vac_id);
        sim = init.sim;
    }

    // Initialize the particle physics data
    {
        ParticleTrackView particle(
            params_.particles, states_.particles, vac_id);
        particle = init.particle;
    }

    // Initialize the geometry
    {
        GeoTrackView geo(params_.geometry, states_.geometry, vac_id);
        if (tid < data_.parents.size())
        {
            // Copy the geometry state from the parent for improved
            // performance
            ThreadId parent_id
                = data_.parents[from_back(data_.parents.size(), tid)];
            GeoTrackView parent(params_.geometry, states_.geometry, parent_id);
            geo = GeoTrackView::DetailedInitializer{parent, init.geo.dir};
        }
        else
        {
            // Initialize it from the position (more expensive)
            geo = init.geo;
        }

        // Initialize the material
        GeoMaterialView   geo_mat(params_.geo_mats);
        MaterialTrackView mat(params_.materials, states_.materials, vac_id);
        mat = {geo_mat.material_id(geo.volume_id())};
    }

    // Initialize the physics state
    {
        PhysicsTrackView phys(params_.physics, states_.physics, {}, {}, vac_id);
        phys = {};
    }

    // Interaction representing creation of a new track
    {
        states_.interactions[vac_id].action = Action::spawned;
    }
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
