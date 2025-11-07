//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/TrackInitAlgorithms.cu
//---------------------------------------------------------------------------//
#include "TrackInitAlgorithms.hh"

// CUDA has included cub since CUDA 11, but ROCm does not include hipCUB by
// default, so test for the availability of hipCUB and use thrust instead if
// it's unavailable. And some further checks for newer cub/hipCUB functions.
#if CELER_USE_HIP && !CELERITAS_HAVE_HIPCUB
#    define CELER_USE_THRUST 1
#else
#    define CELER_USE_THRUST 0
#endif
#if CELERITAS_USE_CUDA
#    include <cub/device/device_partition.cuh>
#    include <cub/device/device_scan.cuh>
#    include <cub/device/device_select.cuh>
#    include <cub/version.cuh>
#elif CELERITAS_USE_HIP && CELERITAS_HAVE_HIPCUB
#    include <hipcub/device/device_partition.hpp>
#    include <hipcub/device/device_scan.hpp>
#    include <hipcub/device/device_select.hpp>
#    include <hipcub/hipcub_version.hpp>
#endif
// DeviceTransform is unavailable in older versions of cub/hipcub, so fall back
// to using thrust::transform instead
#if CELERITAS_USE_CUDA && CUB_VERSION >= 200800
#    define CELER_CUB_HAS_TRANSFORM 1
#else
#    define CELER_CUB_HAS_TRANSFORM 0
#endif
#if CELERITAS_USE_HIP && HIPCUB_VERSION >= 400100
#    define CELER_HIPCUB_HAS_TRANSFORM 1
#else
#    define CELER_HIPCUB_HAS_TRANSFORM 0
#endif
#if CELER_CUB_HAS_TRANSFORM
#    include <cub/device/device_transform.cuh>
#elif CELER_HIPCUB_HAS_TRANSFORM
#    include <hipcub/device/device_transform.hpp>
#else
#    include <thrust/execution_policy.h>
#    include <thrust/transform.h>
#endif
#include <thrust/device_ptr.h>
#if CELER_USE_THRUST
#    include <thrust/partition.h>
#    include <thrust/remove.h>
#    include <thrust/scan.h>
#endif

#include "corecel/Macros.hh"
#include "corecel/data/DeviceVector.hh"
#include "corecel/data/ObserverPtr.device.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/Stream.hh"
#include "corecel/sys/Thrust.device.hh"

#include "../Utils.hh"

#if CELERITAS_USE_HIP && !CELERITAS_HAVE_HIPCUB
namespace cub = hipcub;
#endif

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Whether the track slot is being used
 */
struct NotNull
{
    CELER_FUNCTION bool operator()(TrackSlotId a) const noexcept
    {
        return a.get() != TrackSlotId{}.unchecked_get();
    }
};

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
#if CELER_USE_THRUST
    auto start = device_pointer_cast(vacancies.data());
    auto end = thrust::remove_if(thrust_execute_on(stream_id),
                                 start,
                                 start + vacancies.size(),
                                 LogicalNot{});
    CELER_DEVICE_API_CALL(PeekAtLastError());

    // New size of the vacancy vector
    return end - start;
#else
    // cub functions expect a cudaStream_t pointer for the stream
    using StreamT = CELER_DEVICE_API_SYMBOL(Stream_t);
    StreamT stream = device().stream(stream_id).get();

    DeviceVector<size_type> num_not_active{1, stream_id};

    // Calling with nullptr causes the function to return the amount of working
    // space needed instead of invoking the kernel.
    size_t temp_storage_bytes = 0;
    auto data = device_pointer_cast(vacancies.data());
    // HIP defines hipCUB functions as [[nodiscard]], but we defer error checks
    auto cub_error_code = cub::DeviceSelect::If(nullptr,
                                                temp_storage_bytes,
                                                data,
                                                num_not_active.data(),
                                                vacancies.size(),
                                                NotNull{},
                                                stream);
    // Allocate temporary storage
    DeviceAllocation temp_storage(temp_storage_bytes, stream_id);
    // Run selection
    cub_error_code = cub::DeviceSelect::If(temp_storage.data(),
                                           temp_storage_bytes,
                                           data,
                                           num_not_active.data(),
                                           vacancies.size(),
                                           NotNull{},
                                           stream);

    auto num = ItemCopier<size_type>{stream_id}(num_not_active.data());

    CELER_DEVICE_API_CALL(PeekAtLastError());
    return num;
#endif
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
#if CELER_USE_THRUST
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
#else
    // cub functions expect a cudaStream_t pointer for the stream
    using StreamT = CELER_DEVICE_API_SYMBOL(Stream_t);
    StreamT stream = device().stream(stream_id).get();

    // Calling with nullptr causes the function to return the amount of working
    // space needed instead of invoking the kernel.
    size_t temp_storage_bytes = 0;
    auto data = device_pointer_cast(counts.data());
    // HIP defines hipCUB functions as [[nodiscard]], but we defer error checks
    auto cub_error_code = cub::DeviceScan::ExclusiveSum(
        nullptr, temp_storage_bytes, data, counts.size(), stream);
    // Allocate temporary storage
    DeviceAllocation temp_storage(temp_storage_bytes, stream_id);
    // Run exclusive prefix sum
    cub_error_code = cub::DeviceScan::ExclusiveSum(
        temp_storage.data(), temp_storage_bytes, data, counts.size(), stream);
    // Set the counter similar to the following
    // counters.num_secondaries = "last value in the counts object;
    CELER_DEVICE_API_CALL(PeekAtLastError());
    return ItemCopier<size_type>{stream_id}(data.get() + counts.size() - 1);
#endif
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
    CELER_EXPECT(count != 0);

    ScopedProfiling profile_this{"partition-initializers"};
#if CELER_USE_THRUST
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
#else
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
#    if CELER_CUB_HAS_TRANSFORM || CELER_HIPCUB_HAS_TRANSFORM
    // HIP defines hipCUB functions as [[nodiscard]], but we defer error checks
    {
        auto cub_error_code = cub::DeviceTransform::Transform(
            stencil,
            flags.data(),
            count,
            IsNeutral{params.ptr<MemSpace::native>()},
            stream);
    }
#    else
    thrust::transform(thrust_execute_on(stream_id),
                      stencil,
                      stencil + count,
                      flags.data(),
                      IsNeutral{params.ptr<MemSpace::native>()});
#    endif
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
    auto cub_error_code = cub::DevicePartition::Flagged(nullptr,
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
    cub_error_code = cub::DevicePartition::Flagged(temp_storage.data(),
                                                   temp_storage_bytes,
                                                   start,
                                                   flags.data(),
                                                   data,
                                                   num_neutral.data(),
                                                   count,
                                                   stream);
    CELER_DEVICE_API_CALL(PeekAtLastError());
#endif
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
