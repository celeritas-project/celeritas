//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file InitializeTracks.cc
//---------------------------------------------------------------------------//
#include "InitializeTracks.hh"

#include <numeric>
#include "base/Atomics.hh"
#include "base/Range.hh"
#include "base/Types.hh"
#include "geometry/GeoMaterialView.hh"
#include "geometry/GeoTrackView.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/PhysicsTrackView.hh"
#include "physics/material/MaterialTrackView.hh"
#include "sim/SimTrackView.hh"
#include "sim/TrackInitData.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
// HELPER CLASSES
//---------------------------------------------------------------------------//
struct IsEqual
{
    size_type value;

    CELER_FUNCTION bool operator()(size_type x) const { return x == value; }
};
} // namespace

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
/*!
 * Find empty slots in the track vector and count the number of secondaries
 * that survived cutoffs for each interaction. If the track is dead and
 * produced secondaries, fill the empty track slot with one of the secondaries.
 */
void locate_alive(const ParamsHostRef&         params,
                  const StateHostRef&          states,
                  const TrackInitStateHostRef& inits)
{
    for (auto tid : range(ThreadId{states.size()}))
    {
        // Index of the secondary to copy to the parent track if vacant
        size_type secondary_idx = flag_id();

        // Count how many secondaries survived cutoffs for each track
        inits.secondary_counts[tid] = 0;
        Interaction& result         = states.interactions[tid];
        for (auto i : range(result.secondaries.size()))
        {
            if (result.secondaries[i])
            {
                if (secondary_idx == flag_id())
                {
                    secondary_idx = i;
                }
                ++inits.secondary_counts[tid];
            }
        }

        SimTrackView sim(states.sim, tid);
        if (sim.alive())
        {
            // The track is alive: mark this track slot as occupied
            inits.vacancies[tid] = flag_id();
        }
        else if (secondary_idx != flag_id())
        {
            // The track was killed and it produced secondaries: fill the empty
            // track slot with the first secondary and mark as occupied

            // Calculate the track ID of the secondary
            // TODO: This is nondeterministic; we need to calculate the track
            // ID in a reproducible way.
            CELER_ASSERT(sim.event_id() < inits.track_counters.size());
            TrackId::size_type track_id
                = atomic_add(&inits.track_counters[sim.event_id()], 1u);

            // Initialize the simulation state
            sim = {TrackId{track_id}, sim.track_id(), sim.event_id(), true};

            // Initialize the particle state from the secondary
            Secondary&        secondary = result.secondaries[secondary_idx];
            ParticleTrackView particle(params.particles, states.particles, tid);
            particle = {secondary.particle_id, secondary.energy};

            // Keep the parent's geometry state but get the direction from the
            // secondary. The material state will be the same as the parent's.
            GeoTrackView geo(params.geometry, states.geometry, tid);
            geo = {geo, secondary.direction};

            // Initialize the physics state
            PhysicsTrackView phys(params.physics, states.physics, {}, {}, tid);
            phys = {};

            // Mark the secondary as processed and the track as active
            --inits.secondary_counts[tid];
            secondary            = Secondary{};
            inits.vacancies[tid] = flag_id();
        }
        else
        {
            // The track was killed and did not produce secondaries: store the
            // index so it can be used later to initialize a new track
            inits.vacancies[tid] = tid.get();
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Remove all elements in the vacancy vector that were flagged as active
 * tracks.
 */
template<>
size_type remove_if_alive<MemSpace::host>(Span<size_type> vacancies)
{
    auto end = std::remove_if(vacancies.data(),
                              vacancies.data() + vacancies.size(),
                              IsEqual{flag_id()});

    // New size of the vacancy vector
    size_type result = end - vacancies.data();
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Sum the total number of surviving secondaries.
 */
template<>
size_type reduce_counts<MemSpace::host>(Span<size_type> counts)
{
    return std::accumulate(counts.begin(), counts.end(), size_type(0));
}

//---------------------------------------------------------------------------//
/*!
 * Do an exclusive scan of the number of surviving secondaries from each track.
 *
 * For an input array x, this calculates the exclusive prefix sum y of the
 * array elements, i.e., \f$ y_i = \sum_{j=0}^{i-1} x_j \f$,
 * where \f$ y_0 = 0 \f$, and stores the result in the input array.
 */
template<>
void exclusive_scan_counts<MemSpace::host>(Span<size_type> counts)
{
    // TODO: Use std::exclusive_scan when C++17 is adopted
    size_type acc = 0;
    for (auto& count_i : counts)
    {
        size_type current = count_i;
        count_i           = acc;
        acc += current;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Create track initializers on device from secondary particles.
 */
void process_secondaries(const ParamsHostRef&         params,
                         const StateHostRef&          states,
                         const TrackInitStateHostRef& inits)
{
    for (auto tid : range(ThreadId{states.size()}))
    {
        // Construct the state accessors
        GeoTrackView geo(params.geometry, states.geometry, tid);
        SimTrackView sim(states.sim, tid);

        // Offset in the vector of track initializers
        size_type offset_id = inits.secondary_counts[tid];

        Interaction& result = states.interactions[tid];
        for (const auto& secondary : result.secondaries)
        {
            if (secondary)
            {
                // The secondary survived cutoffs: convert to a track
                CELER_ASSERT(offset_id < inits.parents.size());
                TrackInitializer& init = inits.initializers[ThreadId(
                    inits.initializers.size() - inits.parents.size() + offset_id)];

                // Store the thread ID of the secondary's parent
                inits.parents[ThreadId{offset_id++}] = tid;

                // Calculate the track ID of the secondary
                // TODO: This is nondeterministic; we need to calculate the
                // track ID in a reproducible way.
                CELER_ASSERT(sim.event_id() < inits.track_counters.size());
                TrackId::size_type track_id
                    = atomic_add(&inits.track_counters[sim.event_id()], 1u);

                // Construct a track initializer from a secondary
                init.sim.track_id         = TrackId{track_id};
                init.sim.parent_id        = sim.track_id();
                init.sim.event_id         = sim.event_id();
                init.sim.alive            = true;
                init.geo.pos              = geo.pos();
                init.geo.dir              = secondary.direction;
                init.particle.particle_id = secondary.particle_id;
                init.particle.energy      = secondary.energy;
            }
        }
        // Clear the interaction
        result = Interaction::from_processed();
    }
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
