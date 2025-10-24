//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/TrackInitAlgorithms.cu
//---------------------------------------------------------------------------//
#include "TrackInitAlgorithms.hh"

#if CELERITAS_USE_CUDA
#    include <cub/device/device_scan.cuh>
#    include <cub/device/device_select.cuh>
#elif CELERITAS_USE_HIP
#    include <hipcub/device/device_scan.hpp>
#    include <hipcub/device/device_select.hpp>
#endif
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/partition.h>
// #include <thrust/remove.h>
// #include <thrust/scan.h>

#include "corecel/DeviceRuntimeApi.hh"

#include "corecel/Macros.hh"
#include "corecel/data/DeviceVector.hh"
#include "corecel/data/ObserverPtr.device.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/Stream.hh"
#include "corecel/sys/Thrust.device.hh"

#include "../Utils.hh"

#if CELERITAS_USE_CUDA
using namespace cub;
#elif CELERITAS_USE_HIP
using namespace hipcub;
#endif

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Remove all elements in the vacancy vector that were flagged as active
 * tracks.
 */

// Create a functor to omit the active tracks
//
struct NotEqual
{
    int compare_;

    NotEqual(int compare) : compare_(compare) {}

    CELER_FUNCTION bool operator()(TrackSlotId const& a) const noexcept
    {
        return (a.get() != compare_);
    }
};

size_type remove_if_alive(
    StateCollection<TrackSlotId, Ownership::reference, MemSpace::device> const&
        vacancies,
    StreamId stream_id)
{
    ScopedProfiling profile_this{"remove-if-alive"};
    using StreamT = CELER_DEVICE_API_SYMBOL(Stream_t);
    Stream& s = device().stream(stream_id);
    StreamT stream = s.get();
    // There should be a better way to initialize this function than the hard
    // coded invalid value
    NotEqual select_op(-1);

    // The first call just computes the number of additional bytes needed for
    // the in-place selection. Calling with nullptr causes this instead of
    // invoking the kernel.
    size_t temp_storage_bytes = 0;
    // *** For testing with existing code for consistency
    // *** Instead, we should place the result in the appropriate GPU variable
    // *** For testing with existing code for consistency

    // *** Allocate temporary storage
    // *** For testing with existing code for consistency
    DeviceVector<size_type> num_not_active{1, stream_id};
    auto data = device_pointer_cast(vacancies.data());
    // *** For testing with existing code for consistency

    // *** First call and memory allocation/deallocation for temp space should
    // *** probably go in ExtendFromSecondariesAction.cu::begin_run()
    // *** Put the exclusive sum allocation there, too?
    DeviceSelect::If(nullptr,
                     temp_storage_bytes,
                     data,
                     num_not_active.data(),
                     vacancies.size(),
                     select_op,
                     stream);

    // Allocate temporary storage
    void* d_temp_storage = s.malloc_async(temp_storage_bytes);

    // Run selection
    DeviceSelect::If(d_temp_storage,
                     temp_storage_bytes,
                     data,
                     num_not_active.data(),
                     vacancies.size(),
                     select_op,
                     stream);

    // Deallocate temporary storage
    s.free_async(d_temp_storage);

    // *** For testing with existing code for consistency
    // *** Replace with the num_vacancies counter
    auto num = ItemCopier<size_type>{stream_id}(num_not_active.data());

    // *** For testing with existing code for consistency

    CELER_DEVICE_API_CALL(PeekAtLastError());
    // New size of the vacancy vector
    return num;
    // ScopedProfiling profile_this{"remove-if-alive"};
    // auto start = device_pointer_cast(vacancies.data());
    // auto end = thrust::remove_if(thrust_execute_on(stream_id),
    // start,
    // start + vacancies.size(),
    // LogicalNot{});
    // CELER_DEVICE_API_CALL(PeekAtLastError());

    // CELER_LOG(info) << "thrust found and released " << end - start << "
    // active track slots";

    // // New size of the vacancy vector
    // return end - start;
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
    // To Do:
    // (1) Store the result in the GPU variable that needs the result
    // (2) See (1) and change to void function

    // The first call just computes the number of additional bytes needed for
    // the in-place selection. Calling with nullptr causes this instead of
    // invoking the kernel.

    using StreamT = CELER_DEVICE_API_SYMBOL(Stream_t);
    Stream& s = device().stream(stream_id);
    StreamT stream = s.get();

    size_t temp_storage_bytes = 0;
    auto data = device_pointer_cast(counts.data());
    DeviceScan::ExclusiveSum(
        nullptr, temp_storage_bytes, data, counts.size(), stream);

    // Allocate temporary storage
    void* d_temp_storage = s.malloc_async(temp_storage_bytes);

    // Run exclusive prefix sum
    DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes, data, counts.size(), stream);

    // Deallocate temporary storage
    s.free_async(d_temp_storage);

    // Set the counter similar to the following
    // counters.num_secondaries = ItemCopier<size_type>{stream_id}...;
    CELER_DEVICE_API_CALL(PeekAtLastError());
    return ItemCopier<size_type>{stream_id}(counts.data().get() + counts.size()
                                            - 1);
    // ScopedProfiling profile_this{"exclusive-scan-counts"};
    // // Exclusive scan:
    // auto data = device_pointer_cast(counts.data());
    // auto stop = thrust::exclusive_scan(thrust_execute_on(stream_id),
    // data,
    // data + counts.size(),
    // data,
    // size_type(0));
    // CELER_DEVICE_API_CALL(PeekAtLastError());

    // // Copy the last element (accumulated total) back to host
    // return ItemCopier<size_type>{stream_id}(stop.get() - 1);
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
