//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file InitializeTracks.cu
//---------------------------------------------------------------------------//
#include "InitializeTracks.hh"

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <vector>
#include "base/Atomics.hh"
#include "base/DeviceVector.hh"
#include "base/KernelParamCalculator.cuda.hh"
#include "geometry/GeoMaterialView.hh"
#include "geometry/GeoTrackView.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/PhysicsTrackView.hh"
#include "physics/material/MaterialTrackView.hh"
#include "sim/SimTrackView.hh"

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

//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
/*!
 * Initialize the track states on device. The track initializers are created
 * from either primary particles or secondaries. The new tracks are inserted
 * into empty slots (vacancies) in the track vector.
 */
__global__ void init_tracks_kernel(const ParamsDeviceRef         params,
                                   const StateDeviceRef          states,
                                   const TrackInitStateDeviceRef inits,
                                   size_type                     num_vacancies)
{
    auto tid = KernelParamCalculator::thread_id();
    if (!(tid < num_vacancies))
        return;

    // Get the track initializer from the back of the vector. Since new
    // initializers are pushed to the back of the vector, these will be the
    // most recently added and therefore the ones that still might have a
    // parent they can copy the geometry state from.
    const TrackInitializer& init
        = inits.initializers[from_back(inits.initializers.size(), tid)];

    // Thread ID of vacant track where the new track will be initialized
    ThreadId vac_id(inits.vacancies[from_back(inits.vacancies.size(), tid)]);

    // Initialize the simulation state
    {
        SimTrackView sim(states.sim, vac_id);
        sim = init.sim;
    }

    // Initialize the particle physics data
    {
        ParticleTrackView particle(params.particles, states.particles, vac_id);
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
            GeoTrackView parent(params.geometry, states.geometry, parent_id);
            geo = {parent, init.geo.dir};
        }
        else
        {
            // Initialize it from the position (more expensive)
            geo = init.geo;
        }

        // Initialize the material
        GeoMaterialView   geo_mat(params.geo_mats, geo.volume_id());
        MaterialTrackView mat(params.materials, states.materials, vac_id);
        mat = {geo_mat.material_id()};
    }

    // Initialize the physics state
    {
        PhysicsTrackView phys(params.physics, states.physics, {}, {}, vac_id);
        phys = {};
    }

    // Interaction representing creation of a new track
    {
        Interaction& result = states.interactions[vac_id];
        result.action       = Action::spawned;
        result.energy       = init.particle.energy;
        result.direction    = init.geo.dir;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Find empty slots in the track vector and count the number of secondaries
 * that survived cutoffs for each interaction. If the track is dead and
 * produced secondaries, fill the empty track slot with one of the secondaries.
 */
__global__ void locate_alive_kernel(const ParamsDeviceRef         params,
                                    const StateDeviceRef          states,
                                    const TrackInitStateDeviceRef inits)
{
    auto tid = KernelParamCalculator::thread_id();
    if (!(tid < states.size()))
        return;

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

        // Interaction representing creation of a new track
        Interaction& result = states.interactions[tid];
        result.action       = Action::spawned;
        result.energy       = secondary.energy;
        result.direction    = secondary.direction;

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

//---------------------------------------------------------------------------//
/*!
 * Create track initializers on device from primary particles.
 */
__global__ void process_primaries_kernel(const Span<const Primary> primaries,
                                         const TrackInitStateDeviceRef inits)
{
    auto tid = KernelParamCalculator::thread_id();
    if (!(tid < primaries.size()))
        return;

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

//---------------------------------------------------------------------------//
/*!
 * Create track initializers on device from secondary particles.
 */
__global__ void process_secondaries_kernel(const ParamsDeviceRef params,
                                           const StateDeviceRef  states,
                                           const TrackInitStateDeviceRef inits)
{
    auto tid = KernelParamCalculator::thread_id();
    if (!(tid < states.size()))
        return;

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
    // Clear the secondaries from the interaction
    result.secondaries = {};
}
} // end namespace

//---------------------------------------------------------------------------//
// KERNEL INTERFACE
//---------------------------------------------------------------------------//
/*!
 * Initialize the track states on device.
 */
void init_tracks(const ParamsDeviceRef&         params,
                 const StateDeviceRef&          states,
                 const TrackInitStateDeviceRef& inits)
{
    // Number of vacancies, limited by the initializer size
    auto num_vacancies
        = std::min(inits.vacancies.size(), inits.initializers.size());

    // Initialize tracks on device
    static const celeritas::KernelParamCalculator calc_launch_params(
        init_tracks_kernel, "init_tracks");
    auto lparams = calc_launch_params(num_vacancies);
    init_tracks_kernel<<<lparams.grid_size, lparams.block_size>>>(
        params, states, inits, num_vacancies);
    CELER_CUDA_CHECK_ERROR();
}

//---------------------------------------------------------------------------//
/*!
 * Find empty slots in the vector of tracks and count the number of secondaries
 * that survived cutoffs for each interaction.
 */
void locate_alive(const ParamsDeviceRef&         params,
                  const StateDeviceRef&          states,
                  const TrackInitStateDeviceRef& inits)
{
    static const celeritas::KernelParamCalculator calc_launch_params(
        locate_alive_kernel, "locate_alive");
    auto lparams = calc_launch_params(states.size());
    locate_alive_kernel<<<lparams.grid_size, lparams.block_size>>>(
        params, states, inits);
    CELER_CUDA_CHECK_ERROR();
}

//---------------------------------------------------------------------------//
/*!
 * Create track initializers from primary particles.
 */
void process_primaries(Span<const Primary>            primaries,
                       const TrackInitStateDeviceRef& inits)
{
    CELER_EXPECT(primaries.size() <= inits.initializers.size());

    static const celeritas::KernelParamCalculator calc_launch_params(
        process_primaries_kernel, "process_primaries");
    auto lparams = calc_launch_params(primaries.size());
    process_primaries_kernel<<<lparams.grid_size, lparams.block_size>>>(
        primaries, inits);
    CELER_CUDA_CHECK_ERROR();
}

//---------------------------------------------------------------------------//
/*!
 * Create track initializers from secondary particles.
 */
void process_secondaries(const ParamsDeviceRef&         params,
                         const StateDeviceRef&          states,
                         const TrackInitStateDeviceRef& inits)
{
    CELER_EXPECT(states.size() <= inits.secondary_counts.size());
    CELER_EXPECT(states.size() <= states.interactions.size());

    static const celeritas::KernelParamCalculator calc_launch_params(
        process_secondaries_kernel, "process_secondaries");
    auto lparams = calc_launch_params(states.size());
    process_secondaries_kernel<<<lparams.grid_size, lparams.block_size>>>(
        params, states, inits);
    CELER_CUDA_CHECK_ERROR();
}

//---------------------------------------------------------------------------//
/*!
 * Remove all elements in the vacancy vector that were flagged as active
 * tracks.
 */
size_type remove_if_alive(Span<size_type> vacancies)
{
    thrust::device_ptr<size_type> end = thrust::remove_if(
        thrust::device_pointer_cast(vacancies.data()),
        thrust::device_pointer_cast(vacancies.data() + vacancies.size()),
        IsEqual{flag_id()});

    CELER_CUDA_CHECK_ERROR();

    // New size of the vacancy vector
    size_type result = thrust::raw_pointer_cast(end) - vacancies.data();
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Sum the total number of surviving secondaries.
 */
size_type reduce_counts(Span<size_type> counts)
{
    size_type result = thrust::reduce(
        thrust::device_pointer_cast(counts.data()),
        thrust::device_pointer_cast(counts.data()) + counts.size(),
        size_type(0),
        thrust::plus<size_type>());

    CELER_CUDA_CHECK_ERROR();
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Do an exclusive scan of the number of surviving secondaries from each track.
 *
 * For an input array x, this calculates the exclusive prefix sum y of the
 * array elements, i.e., \f$ y_i = \sum_{j=0}^{i-1} x_j \f$,
 * where \f$ y_0 = 0 \f$, and stores the result in the input array.
 */
void exclusive_scan_counts(Span<size_type> counts)
{
    thrust::exclusive_scan(
        thrust::device_pointer_cast(counts.data()),
        thrust::device_pointer_cast(counts.data()) + counts.size(),
        counts.data(),
        size_type(0));

    CELER_CUDA_CHECK_ERROR();
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
