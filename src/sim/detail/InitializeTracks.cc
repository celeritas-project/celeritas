//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file InitializeTracks.cc
//---------------------------------------------------------------------------//
#include "InitializeTracks.hh"

#include "base/Range.hh"
#include "base/Types.hh"
#include "geometry/GeoMaterialView.hh"
#include "geometry/GeoTrackView.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/PhysicsTrackView.hh"
#include "physics/material/MaterialTrackView.hh"
#include "sim/SimTrackView.hh"
#include "sim/TrackInitInterface.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Create track initializers on device from primary particles.
 */
void process_primaries(Span<const Primary>          primaries,
                       const TrackInitStateHostRef& inits)
{
    // TODO: What to do about celeritas::size_type for host?
    for (auto tid :
         range(ThreadId{static_cast<celeritas::size_type>(primaries.size())}))
    {
        TrackInitializer& init    = inits.initializers[ThreadId(
            inits.initializers.size() - primaries.size() + tid.get())];
        const Primary&    primary = primaries[tid.get()];

        // Construct a track initializer from a primary particle
        init.sim.track_id         = primary.track_id;
        init.sim.parent_id        = TrackId{};
        init.sim.event_id         = primary.event_id;
        init.sim.alive            = true;
        init.geo.pos              = primary.position;
        init.geo.dir              = primary.direction;
        init.particle.particle_id = primary.particle_id;
        init.particle.energy      = primary.energy;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Initialize the track states on device. The track initializers are created
 * from either primary particles or secondaries. The new tracks are inserted
 * into empty slots (vacancies) in the track vector.
 */
void init_tracks(const ParamsHostRef&         params,
                 const StateHostRef&          states,
                 const TrackInitStateHostRef& inits)
{
    // Number of vacancies, limited by the initializer size
    auto num_vacancies
        = std::min(inits.vacancies.size(), inits.initializers.size());

    for (auto tid : range(ThreadId{num_vacancies}))
    {
        // Get the track initializer from the back of the vector. Since new
        // initializers are pushed to the back of the vector, these will be the
        // most recently added and therefore the ones that still might have a
        // parent they can copy the geometry state from.
        const TrackInitializer& init
            = inits.initializers[from_back(inits.initializers.size(), tid)];

        // Thread ID of vacant track where the new track will be initialized
        ThreadId vac_id(
            inits.vacancies[from_back(inits.vacancies.size(), tid)]);

        // Initialize the simulation state
        {
            SimTrackView sim(states.sim, vac_id);
            sim = init.sim;
        }

        // Initialize the particle physics data
        {
            ParticleTrackView particle(
                params.particles, states.particles, vac_id);
            particle = init.particle;
        }

        // Initialize the geometry
        {
            GeoTrackView geo(params.geometry, states.geometry, vac_id);
            if (tid < inits.parents.size())
            {
                // Copy the geometry state from the parent for improved
                // performance
                ThreadId parent_id
                    = inits.parents[from_back(inits.parents.size(), tid)];
                GeoTrackView parent(
                    params.geometry, states.geometry, parent_id);
                geo = {parent, init.geo.dir};
            }
            else
            {
                // Initialize it from the position (more expensive)
                geo = init.geo;
            }

            // Initialize the material
            GeoMaterialView   geo_mat(params.geo_mats);
            MaterialTrackView mat(params.materials, states.materials, vac_id);
            mat = {geo_mat.material_id(geo.volume_id())};
        }

        // Initialize the physics state
        {
            PhysicsTrackView phys(
                params.physics, states.physics, {}, {}, vac_id);
            phys = {};
        }

        // Interaction representing creation of a new track
        {
            states.interactions[vac_id].action = Action::spawned;
        }
    }
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
