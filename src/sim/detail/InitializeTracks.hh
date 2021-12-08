//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file InitializeTracks.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Atomics.hh"
#include "base/NumericLimits.hh"
#include "base/Span.hh"
#include "geometry/GeoMaterialView.hh"
#include "geometry/GeoTrackView.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/Primary.hh"
#include "physics/base/PhysicsTrackView.hh"
#include "physics/material/MaterialTrackView.hh"
#include "sim/TrackData.hh"
#include "sim/TrackInitData.hh"
#include "sim/SimTrackView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// HELPER CLASSES
//---------------------------------------------------------------------------//
struct IsEqual
{
    size_type value;

    CELER_FUNCTION bool operator()(size_type x) const { return x == value; }
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
//! Invalid index flag
CELER_CONSTEXPR_FUNCTION size_type flag_id()
{
    return numeric_limits<size_type>::max();
}

//---------------------------------------------------------------------------//
//! Get the thread ID of the last element
CELER_FORCEINLINE_FUNCTION ThreadId from_back(size_type size, ThreadId tid)
{
    CELER_EXPECT(tid.get() + 1 <= size);
    return ThreadId{size - tid.get() - 1};
}

//---------------------------------------------------------------------------//
// Initialize track states
void init_tracks(const ParamsDeviceRef&         params,
                 const StateDeviceRef&          states,
                 const TrackInitStateDeviceRef& data);
void init_tracks(const ParamsHostRef&         params,
                 const StateHostRef&          states,
                 const TrackInitStateHostRef& data);

//---------------------------------------------------------------------------//
// Identify which tracks are alive and count secondaries created
void locate_alive(const ParamsDeviceRef&         params,
                  const StateDeviceRef&          states,
                  const TrackInitStateDeviceRef& data);
void locate_alive(const ParamsHostRef&         params,
                  const StateHostRef&          states,
                  const TrackInitStateHostRef& data);

//---------------------------------------------------------------------------//
// Create track initializers from primary particles
void process_primaries(Span<const Primary>            primaries,
                       const TrackInitStateDeviceRef& data);
void process_primaries(Span<const Primary>          primaries,
                       const TrackInitStateHostRef& data);

//---------------------------------------------------------------------------//
// Create track initializers from secondary particles.
void process_secondaries(const ParamsDeviceRef&         params,
                         const StateDeviceRef&          states,
                         const TrackInitStateDeviceRef& data);
void process_secondaries(const ParamsHostRef&         params,
                         const StateHostRef&          states,
                         const TrackInitStateHostRef& data);

//---------------------------------------------------------------------------//
// Remove all elements in the vacancy vector that were flagged as alive
template<MemSpace M>
size_type remove_if_alive(Span<size_type> vacancies);

template<>
size_type remove_if_alive<MemSpace::host>(Span<size_type> vacancies);
template<>
size_type remove_if_alive<MemSpace::device>(Span<size_type> vacancies);

//---------------------------------------------------------------------------//
// Sum the total number of surviving secondaries.
template<MemSpace M>
size_type reduce_counts(Span<size_type> counts);

template<>
size_type reduce_counts<MemSpace::host>(Span<size_type> counts);
template<>
size_type reduce_counts<MemSpace::device>(Span<size_type> counts);

//---------------------------------------------------------------------------//
// Calculate the exclusive prefix sum of the number of surviving secondaries
template<MemSpace M>
void exclusive_scan_counts(Span<size_type> counts);

template<>
void exclusive_scan_counts<MemSpace::host>(Span<size_type> counts);
template<>
void exclusive_scan_counts<MemSpace::device>(Span<size_type> counts);

//---------------------------------------------------------------------------//
// KERNEL LAUNCHERS
//---------------------------------------------------------------------------//
#define LAUNCHER(NAME)                                                     \
    template<MemSpace M>                                                   \
    class NAME##Launcher                                                   \
    {                                                                      \
      public:                                                              \
        using ParamsDataRef = ParamsData<Ownership::const_reference, M>;   \
        using StateDataRef  = StateData<Ownership::reference, M>;          \
        using TrackInitStateDataRef                                        \
            = TrackInitStateData<Ownership::reference, M>;                 \
                                                                           \
      public:                                                              \
        CELER_FUNCTION NAME##Launcher(const ParamsDataRef&         params, \
                                      const StateDataRef&          states, \
                                      const TrackInitStateDataRef& data)   \
            : params_(params), states_(states), data_(data)                \
        {                                                                  \
            CELER_EXPECT(params_);                                         \
            CELER_EXPECT(states_);                                         \
            CELER_EXPECT(data_);                                           \
        }                                                                  \
                                                                           \
        inline CELER_FUNCTION void operator()(ThreadId tid) const;         \
                                                                           \
      private:                                                             \
        const ParamsDataRef&         params_;                              \
        const StateDataRef&          states_;                              \
        const TrackInitStateDataRef& data_;                                \
    };

LAUNCHER(InitTracks)
LAUNCHER(LocateAlive)
LAUNCHER(ProcessSecondaries)

//---------------------------------------------------------------------------//
/*!
 * Create track initializers from primary particles.
 */
template<MemSpace M>
class ProcessPrimariesLauncher
{
  public:
    //!@{
    //! Type aliases
    using TrackInitStateDataRef = TrackInitStateData<Ownership::reference, M>;
    //!@}

  public:
    // Construct with shared and state data
    CELER_FUNCTION ProcessPrimariesLauncher(Span<const Primary> primaries,
                                            const TrackInitStateDataRef& data)
        : primaries_(primaries), data_(data)
    {
        CELER_EXPECT(data_);
    }

    // Create track initializers from primaries
    inline CELER_FUNCTION void operator()(ThreadId tid) const;

  private:
    Span<const Primary>          primaries_;
    const TrackInitStateDataRef& data_;
};

//---------------------------------------------------------------------------//
// INLINE FUNCTION DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Initialize the track states.
 *
 * The track initializers are created from either primary particles or
 * secondaries. The new tracks are inserted into empty slots (vacancies) in the
 * track vector.
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
/*!
 * Determine which tracks are alive and count secondaries.
 *
 * This finds empty slots in the track vector and counts the number of
 * secondaries created in each interaction. If the track was killed and
 * produced secondaries, the empty track slot is filled with the first
 * secondary.
 */
template<MemSpace M>
CELER_FUNCTION void LocateAliveLauncher<M>::operator()(ThreadId tid) const
{
    // Index of the secondary to copy to the parent track if vacant
    size_type secondary_idx = flag_id();

    // Count how many secondaries survived cutoffs for each track
    data_.secondary_counts[tid] = 0;
    Interaction& result         = states_.interactions[tid];
    for (auto i : range(result.secondaries.size()))
    {
        if (result.secondaries[i])
        {
            if (secondary_idx == flag_id())
            {
                secondary_idx = i;
            }
            ++data_.secondary_counts[tid];
        }
    }

    SimTrackView sim(states_.sim, tid);
    if (sim.alive())
    {
        // The track is alive: mark this track slot as occupied
        data_.vacancies[tid] = flag_id();
    }
    else if (secondary_idx != flag_id())
    {
        // The track was killed and it produced secondaries: fill the empty
        // track slot with the first secondary and mark as occupied

        // Calculate the track ID of the secondary
        // TODO: This is nondeterministic; we need to calculate the track
        // ID in a reproducible way.
        CELER_ASSERT(sim.event_id() < data_.track_counters.size());
        TrackId::size_type track_id
            = atomic_add(&data_.track_counters[sim.event_id()], 1u);

        // Initialize the simulation state
        sim = {TrackId{track_id}, sim.track_id(), sim.event_id(), true};

        // Initialize the particle state from the secondary
        Secondary&        secondary = result.secondaries[secondary_idx];
        ParticleTrackView particle(params_.particles, states_.particles, tid);
        particle = {secondary.particle_id, secondary.energy};

        // Keep the parent's geometry state but get the direction from the
        // secondary. The material state will be the same as the parent's.
        GeoTrackView geo(params_.geometry, states_.geometry, tid);
        geo = GeoTrackView::DetailedInitializer{geo, secondary.direction};

        // Initialize the physics state
        PhysicsTrackView phys(params_.physics, states_.physics, {}, {}, tid);
        phys = {};

        // Mark the secondary as processed and the track as active
        --data_.secondary_counts[tid];
        secondary            = Secondary{};
        data_.vacancies[tid] = flag_id();
    }
    else
    {
        // The track was killed and did not produce secondaries: store the
        // index so it can be used later to initialize a new track
        data_.vacancies[tid] = tid.get();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Create track initializers from primaries.
 */
template<MemSpace M>
CELER_FUNCTION void ProcessPrimariesLauncher<M>::operator()(ThreadId tid) const
{
    TrackInitializer& init    = data_.initializers[ThreadId(
        data_.initializers.size() - primaries_.size() + tid.get())];
    const Primary&    primary = primaries_[tid.get()];

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

//---------------------------------------------------------------------------//
/*!
 * Create track initializers from secondaries.
 */
template<MemSpace M>
CELER_FUNCTION void
ProcessSecondariesLauncher<M>::operator()(ThreadId tid) const
{
    // Construct the state accessors
    GeoTrackView geo(params_.geometry, states_.geometry, tid);
    SimTrackView sim(states_.sim, tid);

    // Offset in the vector of track initializers
    size_type offset_id = data_.secondary_counts[tid];

    Interaction& result = states_.interactions[tid];
    for (const auto& secondary : result.secondaries)
    {
        if (secondary)
        {
            // The secondary survived cutoffs: convert to a track
            CELER_ASSERT(offset_id < data_.parents.size());
            TrackInitializer& init = data_.initializers[ThreadId(
                data_.initializers.size() - data_.parents.size() + offset_id)];

            // Store the thread ID of the secondary's parent
            data_.parents[ThreadId{offset_id++}] = tid;

            // Calculate the track ID of the secondary
            // TODO: This is nondeterministic; we need to calculate the
            // track ID in a reproducible way.
            CELER_ASSERT(sim.event_id() < data_.track_counters.size());
            TrackId::size_type track_id
                = atomic_add(&data_.track_counters[sim.event_id()], 1u);

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

//---------------------------------------------------------------------------//
#undef LAUNCHER
} // namespace detail
} // namespace celeritas
