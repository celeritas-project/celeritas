//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/TrackInitAlgorithms.cu
//---------------------------------------------------------------------------//
#include "TrackInitAlgorithms.hh"

#if CELERITAS_USE_CUDA
#    include <cub/device/device_partition.cuh>
#    include <cub/device/device_scan.cuh>
#    include <cub/device/device_select.cuh>
#    include <cub/version.cuh>
#    if CUB_VERSION >= 200800
#        include <cub/device/device_transform.cuh>
#    else
#        include <thrust/transform.h>
#    endif
#elif CELERITAS_USE_HIP
#    include <hipcub/device/device_partition.cuh>
#    include <hipcub/device/device_scan.hpp>
#    include <hipcub/device/device_select.hpp>
#    include <hipcub/hipcub_version.hpp>
#    if HIPCUB_VERSION >= 400100
#        include <hipcub/device/device_transform.hpp>
#    else
#        include <thrust/transform.h>
#    endif
#endif
#include <thrust/device_ptr.h>
// #include <thrust/execution_policy.h>
// #include <thrust/partition.h>
// #include <thrust/remove.h>
// #include <thrust/scan.h>

#include "corecel/DeviceRuntimeApi.hh"

#include "corecel/Macros.hh"
#include "corecel/data/DeviceVector.hh"
#include "corecel/data/ObserverPtr.device.hh"
// #include "corecel/math/Algorithms.hh" //Use if revert to LogicalNot()
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
 * Create a functor to recognize specific tracks.
 */

//
//
template<class T>
struct NotEqual
{
    int compare_;

    NotEqual<T>(int compare) : compare_(compare) {}

    CELER_FUNCTION bool operator()(T const& a) const noexcept
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
    // cub functions expect a cudaStream_t pointer for the stream
    using StreamT = CELER_DEVICE_API_SYMBOL(Stream_t);
    StreamT stream = device().stream(stream_id).get();
    // There should be a better way to instantiate this functor than the hard
    // coded invalid value. Some way to use the null value?
    NotEqual<TrackSlotId> select_op(-1);

    // To Do:
    // (1) Store the result in the GPU variable that needs the result
    // (2) Change to a void function

    // *** For testing with existing code for consistency
    // *** Instead, we should place the result in the appropriate GPU variable
    DeviceVector<size_type> num_not_active{1, stream_id};
    // *** For testing with existing code for consistency

    // Calling with nullptr causes the function to return the amount of working
    // space needed instead of invoking the kernel.
    size_t temp_storage_bytes = 0;
    auto data = device_pointer_cast(vacancies.data());
    DeviceSelect::If(nullptr,
                     temp_storage_bytes,
                     data,
                     num_not_active.data(),
                     vacancies.size(),
                     select_op,
                     stream);

    // Allocate temporary storage
    DeviceAllocation temp_storage(temp_storage_bytes, stream_id);

    // Run selection
    DeviceSelect::If(temp_storage.data(),
                     temp_storage_bytes,
                     data,
                     num_not_active.data(),
                     vacancies.size(),
                     select_op,
                     stream);

    // *** For testing with existing code for consistency
    // *** Replace with the num_vacancies counter
    auto num = ItemCopier<size_type>{stream_id}(num_not_active.data());
    // *** For testing with existing code for consistency

    CELER_DEVICE_API_CALL(PeekAtLastError());
    return num;
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
    // cub functions expect a cudaStream_t pointer for the stream
    using StreamT = CELER_DEVICE_API_SYMBOL(Stream_t);
    StreamT stream = device().stream(stream_id).get();

    // To Do:
    // (1) Store the result in the GPU variable that needs the result
    // (2) Change to void function

    // Calling with nullptr causes the function to return the amount of working
    // space needed instead of invoking the kernel.
    size_t temp_storage_bytes = 0;
    auto data = device_pointer_cast(counts.data());
    DeviceScan::ExclusiveSum(
        nullptr, temp_storage_bytes, data, counts.size(), stream);

    // Allocate temporary storage
    DeviceAllocation temp_storage(temp_storage_bytes, stream_id);

    // Run exclusive prefix sum
    DeviceScan::ExclusiveSum(
        temp_storage.data(), temp_storage_bytes, data, counts.size(), stream);

    // Set the counter similar to the following
    // counters.num_secondaries = "last value in the counts object;
    CELER_DEVICE_API_CALL(PeekAtLastError());
    return ItemCopier<size_type>{stream_id}(data.get() + counts.size() - 1);
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

    // Understandably, celeritas doesn't like creating zero-byte vectors. Since
    // cub needs some vectors, trying to allocate these leads to a failed
    // assertion. So, just return. No need to partition zero tracks.
    if (count == 0)
        return;

    // cub functions expect a cudaStream_t pointer for the stream
    using StreamT = CELER_DEVICE_API_SYMBOL(Stream_t);
    StreamT stream = device().stream(stream_id).get();

    // cub doesn't have a partition function that allows the user to specify
    // both an iterator for the values to use for selection and a function to
    // operate on that iterator. (This should change in the future.) So,
    // instead we create an iterator by using a functor to transform the
    // stencil values into boolean flags that determine how to partition
    // the indices.

    // The initializers array is large. Use stencil to point to the start where
    // this array is being used
    auto stencil = static_cast<TrackInitializer*>(init.initializers.data())
                   + counters.num_initializers - count;
    DeviceVector<unsigned char> flags{count, stream_id};
    // DeviceTransform added in cub 2.8/hipcub 4.1, else fall back to thrust
#if CELERITAS_USE_CUDA && CUB_VERSION >= 200800
    DeviceTransform::Transform(stencil,
                               flags.data(),
                               count,
                               IsNeutral{params.ptr<MemSpace::native>()},
                               stream);
#elif CELERITAS_USE_HIP && HIPCUB_VERSION >= 400100
    DeviceTransform::Transform(stencil,
                               flags.data(),
                               count,
                               IsNeutral{params.ptr<MemSpace::native>()},
                               stream);
#else
    thrust::transform(thrust_execute_on(stream_id),
                      stencil,
                      stencil + count,
                      flags.data(),
                      IsNeutral{params.ptr<MemSpace::native>()});
#endif
    // cub doesn't support in-place partitioning, so create a new variable,
    // initial, of the same type and copy the current data in the init.indices
    // object. Use initial for the input data and overwrite init.indices with
    // the partitioned data, as expected from an in-place algorithm.
    StateCollection<size_type, Ownership::value, MemSpace::device> initial;
    initial = init.indices;
    // Calling with nullptr causes the function to return the amount of working
    // space needed instead of invoking the kernel.
    size_t temp_storage_bytes = 0;
    auto start = device_pointer_cast(initial.data());
    auto data = device_pointer_cast(init.indices.data());
    // Allocate storage for the number of neutral tracks (unused by celeritas)
    DeviceVector<size_type> num_neutral{1, stream_id};
    DevicePartition::Flagged(nullptr,
                             temp_storage_bytes,
                             start,
                             flags.data(),
                             data,
                             num_neutral.data(),
                             count,
                             stream);

    // Allocate temporary storage
    DeviceAllocation temp_storage(temp_storage_bytes, stream_id);
    // Partition the indices based on the track initializer charge
    DevicePartition::Flagged(temp_storage.data(),
                             temp_storage_bytes,
                             start,
                             flags.data(),
                             data,
                             num_neutral.data(),
                             count,
                             stream);
    CELER_DEVICE_API_CALL(PeekAtLastError());
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
