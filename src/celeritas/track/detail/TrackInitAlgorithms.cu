//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/TrackInitAlgorithms.cu
//---------------------------------------------------------------------------//
#include "TrackInitAlgorithms.hh"

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/partition.h>
#include <thrust/remove.h>
#include <thrust/scan.h>

#include "corecel/Macros.hh"
#include "corecel/data/ObserverPtr.device.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/Stream.hh"
#include "corecel/sys/Thrust.device.hh"

#include "Utils.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Remove all elements in the vacancy vector that were flagged as active
 * tracks.
 */
size_type remove_if_alive(
    StateCollection<TrackSlotId, Ownership::reference, MemSpace::device> const&
        vacancies,
    StreamId stream_id)
{
    ScopedProfiling profile_this{"remove-if-alive"};
    auto start = device_pointer_cast(vacancies.data());
    auto end = thrust::remove_if(thrust_execute_on(stream_id),
                                 start,
                                 start + vacancies.size(),
                                 IsEqual{occupied()});
    CELER_DEVICE_API_CALL(PeekAtLastError());

    // New size of the vacancy vector
    return end - start;
}

//---------------------------------------------------------------------------//
/*!
 * Do an exclusive scan of the number of secondaries produced by each track.
 *
 * For an input array x, this calculates the exclusive prefix sum y of the
 * array elements, i.e., \f$ y_i = \sum_{j=0}^{i-1} x_j \f$,
 * where \f$ y_0 = 0 \f$, and stores the result in the input array.
 *
 * The return value is the sum of all elements in the input array.
 */
size_type exclusive_scan_counts(
    StateCollection<size_type, Ownership::reference, MemSpace::device> const&
        counts,
    StreamId stream_id)
{
    ScopedProfiling profile_this{"exclusive-scan-counts"};
    // Exclusive scan:
    auto data = device_pointer_cast(counts.data());
    auto stop = thrust::exclusive_scan(thrust_execute_on(stream_id),
                                       data,
                                       data + counts.size(),
                                       data,
                                       size_type(0));
    CELER_DEVICE_API_CALL(PeekAtLastError());

    // Copy the last element (accumulated total) back to host
    return ItemCopier<size_type>{stream_id}(stop.get() - 1);
}

//---------------------------------------------------------------------------//
/*!
 * Sort the tracks that will be initialized in this step by charged/neutral.
 *
 * This partitions an array of indices used to access the track initializers
 * and the thread IDs of the initializers' parent tracks.
 */
void partition_initializers(
    CoreParams const& params,
    TrackInitStateData<Ownership::reference, MemSpace::device> const& init,
    CoreStateCounters const& counters,
    size_type count,
    StreamId stream_id)
{
    ScopedProfiling profile_this{"partition-initializers"};

    // Partition the indices based on the track initializer charge
    auto start = device_pointer_cast(init.indices.data());
    auto end = start + count;
    auto stencil = static_cast<TrackInitializer*>(init.initializers.data())
                   + counters.num_initializers - count;
    thrust::stable_partition(
        thrust_execute_on(stream_id),
        start,
        end,
        IsNeutralStencil{params.ptr<MemSpace::native>(), stencil});
    CELER_DEVICE_API_CALL(PeekAtLastError());
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
