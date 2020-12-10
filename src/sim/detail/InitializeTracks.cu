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
#include "geometry/GeoTrackView.hh"
#include "physics/base/ParticleTrackView.hh"
#include "sim/SimTrackView.hh"

namespace
{
using namespace celeritas;
using celeritas::detail::flag_id;
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
__global__ void init_tracks_kernel(const StatePointers            states,
                                   const ParamPointers            params,
                                   const TrackInitializerPointers inits,
                                   size_type num_vacancies)
{
    auto thread_id = KernelParamCalculator::thread_id().get();
    if (thread_id < num_vacancies)
    {
        // Get the track initializer from the back of the vector. Since new
        // initializers are pushed to the back of the vector, these will be the
        // most recently added and therefore the ones that still might have a
        // parent they can copy the geometry state from.
        const TrackInitializer& init
            = inits.initializers[inits.initializers.size() - thread_id - 1];

        // Index of the empty slot to create the new track in
        ThreadId slot_id(
            inits.vacancies[inits.vacancies.size() - thread_id - 1]);

        // Initialize the simulation state
        {
            SimTrackView sim(states.sim, slot_id);
            sim = init.sim;
        }

        // Initialize the particle physics data
        {
            ParticleTrackView particle(
                params.particle, states.particle, slot_id);
            particle = init.particle;
        }

        // Initialize the geometry
        {
            GeoTrackView geo(params.geo, states.geo, slot_id);
            if (thread_id < inits.parent.size())
            {
                // Copy the geometry state from the parent for improved
                // performance
                TrackId::value_type parent_id
                    = inits.parent[inits.parent.size() - thread_id - 1];
                GeoTrackView parent(
                    params.geo, states.geo, ThreadId{parent_id});
                geo = {parent, init.geo.dir};
            }
            else
            {
                // Initialize it from the position (more expensive)
                geo = init.geo;
            }
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Find empty slots in the track vector and count the number of secondaries
 * that survived cutoffs for each interaction. If the track is dead and
 * produced secondaries, fill the empty track slot with one of the secondaries.
 */
__global__ void locate_alive_kernel(const StatePointers            states,
                                    const ParamPointers            params,
                                    const TrackInitializerPointers inits)
{
    auto thread_id = KernelParamCalculator::thread_id();
    if (thread_id < states.size())
    {
        // Secondary to copy to the parent's track slot if the parent has died
        size_type secondary_id = flag_id();

        // Count how many secondaries survived cutoffs for each track
        inits.secondary_counts[thread_id.get()] = 0;
        Interaction& result = states.interactions[thread_id.get()];
        for (size_type i = 0; i < result.secondaries.size(); ++i)
        {
            if (result.secondaries[i])
            {
                if (secondary_id == flag_id())
                {
                    secondary_id = i;
                }
                ++inits.secondary_counts[thread_id.get()];
            }
        }

        SimTrackView sim(states.sim, thread_id);
        if (sim.alive())
        {
            // The track is alive: mark this track slot as active
            inits.vacancies[thread_id.get()] = flag_id();
        }
        else if (secondary_id != flag_id())
        {
            // The track is dead and produced secondaries: fill the empty track
            // slot with the first secondary and mark the track slot as active

            // Calculate the track ID of the secondary
            // TODO: This is nondeterministic; we need to calculate the track
            // ID in a reproducible way.
            CHECK(sim.event_id() < inits.track_counter.size());
            TrackId::value_type track_id
                = atomic_add(&inits.track_counter[sim.event_id().get()], 1u);

            // Initialize the simulation state
            sim = {TrackId{track_id}, sim.track_id(), sim.event_id(), true};

            // Initialize the particle state from the secondary
            Secondary&        secondary = result.secondaries[secondary_id];
            ParticleTrackView particle(
                params.particle, states.particle, thread_id);
            particle = {secondary.def_id, secondary.energy};

            // Keep the parent's geometry state
            GeoTrackView geo(params.geo, states.geo, thread_id);
            geo = {geo, secondary.direction};

            // Mark the secondary as processed and the track as active
            --inits.secondary_counts[thread_id.get()];
            secondary                        = Secondary{};
            inits.vacancies[thread_id.get()] = flag_id();
        }
        else
        {
            // The track is dead and did not produce secondaries: store the
            // index so it can be used later to initialize a new track
            inits.vacancies[thread_id.get()] = thread_id.get();
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Create track initializers on device from primary particles.
 */
__global__ void
process_primaries_kernel(const span<const Primary>    primaries,
                         const span<TrackInitializer> initializers)
{
    auto thread_id = KernelParamCalculator::thread_id();
    if (thread_id < primaries.size())
    {
        TrackInitializer& init    = initializers[thread_id.get()];
        const Primary&    primary = primaries[thread_id.get()];

        // Construct a track initializer from a primary particle
        init.sim.track_id    = primary.track_id;
        init.sim.parent_id   = TrackId{};
        init.sim.event_id    = primary.event_id;
        init.sim.alive       = true;
        init.geo.pos         = primary.position;
        init.geo.dir         = primary.direction;
        init.particle.def_id = primary.def_id;
        init.particle.energy = primary.energy;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Create track initializers on device from secondary particles.
 */
__global__ void process_secondaries_kernel(const StatePointers states,
                                           const ParamPointers params,
                                           const TrackInitializerPointers inits)
{
    auto thread_id = KernelParamCalculator::thread_id();
    if (thread_id < states.size())
    {
        // Construct the state accessors
        GeoTrackView geo(params.geo, states.geo, thread_id);
        SimTrackView sim(states.sim, thread_id);

        // Offset in the vector of track initializers
        size_type offset_id = inits.secondary_counts[thread_id.get()];

        Interaction& result = states.interactions[thread_id.get()];
        for (const auto& secondary : result.secondaries)
        {
            if (secondary)
            {
                // The secondary survived cutoffs: convert to a track
                CHECK(offset_id < inits.initializers.size());
                TrackInitializer& init = inits.initializers[offset_id];

                // Store the thread ID of the secondary's parent
                CHECK(offset_id < inits.parent.size());
                inits.parent[offset_id++] = thread_id.get();

                // Calculate the track ID of the secondary
                // TODO: This is nondeterministic; we need to calculate the
                // track ID in a reproducible way.
                CHECK(sim.event_id() < inits.track_counter.size());
                TrackId::value_type track_id = atomic_add(
                    &inits.track_counter[sim.event_id().get()], 1u);

                // Construct a track initializer from a secondary
                init.sim.track_id    = TrackId{track_id};
                init.sim.parent_id   = sim.track_id();
                init.sim.event_id    = sim.event_id();
                init.sim.alive       = true;
                init.geo.pos         = geo.pos();
                init.geo.dir         = secondary.direction;
                init.particle.def_id = secondary.def_id;
                init.particle.energy = secondary.energy;
            }
        }
        // Clear the secondaries from the interaction
        result.secondaries = {};
    }
}
} // end namespace

//---------------------------------------------------------------------------//

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// KERNEL INTERFACE
//---------------------------------------------------------------------------//
/*!
 * Initialize the track states on device.
 */
void init_tracks(const StatePointers&            states,
                 const ParamPointers&            params,
                 const TrackInitializerPointers& inits)
{
    // Number of vacancies, limited by the initializer size
    auto num_vacancies
        = std::min(inits.vacancies.size(), inits.initializers.size());

    // Initialize tracks on device
    KernelParamCalculator calc_launch_params;
    auto                  lparams = calc_launch_params(num_vacancies);
    init_tracks_kernel<<<lparams.grid_size, lparams.block_size>>>(
        states, params, inits, num_vacancies);

    CELER_CUDA_CALL(cudaDeviceSynchronize());
}

//---------------------------------------------------------------------------//
/*!
 * Find empty slots in the vector of tracks and count the number of secondaries
 * that survived cutoffs for each interaction.
 */
void locate_alive(const StatePointers&            states,
                  const ParamPointers&            params,
                  const TrackInitializerPointers& inits)
{
    KernelParamCalculator calc_launch_params;
    auto                  lparams = calc_launch_params(states.size());
    locate_alive_kernel<<<lparams.grid_size, lparams.block_size>>>(
        states, params, inits);

    CELER_CUDA_CALL(cudaDeviceSynchronize());
}

//---------------------------------------------------------------------------//
/*!
 * Create track initializers from primary particles.
 */
void process_primaries(span<const Primary>             primaries,
                       const TrackInitializerPointers& inits)
{
    REQUIRE(primaries.size() <= inits.initializers.size());

    // Get a view to the last primaries.size() initializers
    auto initializers = inits.initializers.subspan(inits.initializers.size()
                                                   - primaries.size());
    CHECK(initializers.size() == primaries.size());

    KernelParamCalculator calc_launch_params;
    auto                  lparams = calc_launch_params(primaries.size());
    process_primaries_kernel<<<lparams.grid_size, lparams.block_size>>>(
        primaries, initializers);

    CELER_CUDA_CALL(cudaDeviceSynchronize());
}

//---------------------------------------------------------------------------//
/*!
 * Create track initializers from secondary particles.
 */
void process_secondaries(const StatePointers&     states,
                         const ParamPointers&     params,
                         TrackInitializerPointers inits)
{
    REQUIRE(states.size() <= inits.secondary_counts.size());
    REQUIRE(states.size() <= states.interactions.size());

    // Get a view to the last num_secondaries initializers
    inits.initializers = inits.initializers.subspan(inits.initializers.size()
                                                    - inits.parent.size());
    KernelParamCalculator calc_launch_params;
    auto                  lparams = calc_launch_params(states.size());
    process_secondaries_kernel<<<lparams.grid_size, lparams.block_size>>>(
        states, params, inits);

    CELER_CUDA_CALL(cudaDeviceSynchronize());
}

//---------------------------------------------------------------------------//
/*!
 * Remove all elements in the vacancy vector that were flagged as active
 * tracks.
 */
size_type remove_if_alive(span<size_type> vacancies)
{
    thrust::device_ptr<size_type> end = thrust::remove_if(
        thrust::device_pointer_cast(vacancies.data()),
        thrust::device_pointer_cast(vacancies.data() + vacancies.size()),
        IsEqual{flag_id()});

    CELER_CUDA_CALL(cudaDeviceSynchronize());

    // New size of the vacancy vector
    size_type result = thrust::raw_pointer_cast(end) - vacancies.data();
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Sum the total number of surviving secondaries.
 */
size_type reduce_counts(span<size_type> counts)
{
    size_type result = thrust::reduce(
        thrust::device_pointer_cast(counts.data()),
        thrust::device_pointer_cast(counts.data()) + counts.size(),
        size_type(0),
        thrust::plus<size_type>());

    CELER_CUDA_CALL(cudaDeviceSynchronize());
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Do an exclusive scan of the number of surviving secondaries from each track.
 *
 * For an input array x, this calculates the exclusive prefix sum y of the
 * array elements, i.e., y_i = \sum_{j=0}^{i-1} x_j, where y_0 = 0, and stores
 * the result in the input array.
 */
void exclusive_scan_counts(span<size_type> counts)
{
    thrust::exclusive_scan(
        thrust::device_pointer_cast(counts.data()),
        thrust::device_pointer_cast(counts.data()) + counts.size(),
        counts.data(),
        size_type(0));

    CELER_CUDA_CALL(cudaDeviceSynchronize());
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
