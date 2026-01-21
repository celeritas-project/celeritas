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
#    include <thrust/iterator/counting_iterator.h>
#elif CELERITAS_HAVE_HIPCUB
#    include <hipcub/device/device_partition.hpp>
#    include <hipcub/device/device_scan.hpp>
#    include <hipcub/device/device_select.hpp>
#    include <thrust/iterator/counting_iterator.h>
#else
#    include <thrust/execution_policy.h>
#    include <thrust/partition.h>
#    include <thrust/remove.h>
#    include <thrust/scan.h>

#    include "corecel/math/Algorithms.hh"  // For LogicalNot()
#endif
#if CELER_CUB_HAS_TRANSFORM
#    include <cub/device/device_transform.cuh>
#elif CELER_HIPCUB_HAS_TRANSFORM
#    include <hipcub/device/device_transform.hpp>
#elif !CELER_USE_THRUST
#    include <thrust/execution_policy.h>
#    include <thrust/transform.h>
#endif
#include <thrust/device_ptr.h>

#include "corecel/Macros.hh"
#include "corecel/data/DeviceVector.hh"
#include "corecel/data/ObserverPtr.device.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/Stream.hh"
#include "corecel/sys/Thrust.device.hh"

#include "../Utils.hh"

#if CELERITAS_HAVE_HIPCUB
namespace cub = hipcub;
#endif

namespace celeritas
{
namespace detail
{
#if !CELER_USE_THRUST
//---------------------------------------------------------------------------//
/*!
 * Whether the track slot is being used
 */
struct NotNull
{
    CELER_FUNCTION bool operator()(TrackSlotId a) const noexcept
    {
        return static_cast<bool>(a);
    }
};
#endif

//---------------------------------------------------------------------------//
/*!
 * Remove all elements in the vacancy vector that were flagged as active
 * tracks.
 */
void remove_if_alive(
    TrackInitStateData<Ownership::reference, MemSpace::device> const& init,
    StreamId stream_id)
{
    ScopedProfiling profile_this{"remove-if-alive"};
#if CELER_USE_THRUST
    auto& stream = device().stream(stream_id);
    auto start = device_pointer_cast(init.vacancies.data());
    auto counters = device_pointer_cast(init.counters.data());
    auto host_counters
        = ItemCopier<CoreStateCounters>{stream_id}(counters.get());
    auto end = thrust::remove_if(thrust_execute_on(stream_id),
                                 start,
                                 start + init.vacancies.size(),
                                 LogicalNot{});
    CELER_DEVICE_API_CALL(PeekAtLastError());

    // New size of the vacancy vector
    host_counters.num_vacancies = end - start;
    Copier<CoreStateCounters, MemSpace::device> copy{{counters.get(), 1},
                                                     stream_id};
    copy(MemSpace::host, {&host_counters, 1});
    stream.sync();
    return;
#else
    auto& stream = device().stream(stream_id);
    // Calling with nullptr causes the function to return the amount of working
    // space needed instead of invoking the kernel.
    size_t temp_storage_bytes = 0;
    auto data = device_pointer_cast(init.vacancies.data());
    auto counters = device_pointer_cast(init.counters.data());
    // HIP defines hipCUB functions as [[nodiscard]], but we defer error checks
    auto cub_error_code = cub::DeviceSelect::If(nullptr,
                                                temp_storage_bytes,
                                                data,
                                                &(counters->num_vacancies),
                                                init.vacancies.size(),
                                                NotNull{},
                                                stream.get());
    CELER_DISCARD(cub_error_code);
    // Allocate temporary storage
    DeviceVector<char> temp_storage(temp_storage_bytes, stream_id);
    // Run selection
    cub_error_code = cub::DeviceSelect::If(temp_storage.data(),
                                           temp_storage_bytes,
                                           data,
                                           &(counters->num_vacancies),
                                           init.vacancies.size(),
                                           NotNull{},
                                           stream.get());
    CELER_DISCARD(cub_error_code);
    CELER_DEVICE_API_CALL(PeekAtLastError());
    return;
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
    auto& stream = device().stream(stream_id);
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
    auto result = ItemCopier<size_type>{stream_id}(stop.get() - 1);

    stream.sync();
    return result;
#else
    // Calling with nullptr causes the function to return the amount of working
    // space needed instead of invoking the kernel.
    size_t temp_storage_bytes = 0;
    auto data = device_pointer_cast(counts.data());
    // HIP defines hipCUB functions as [[nodiscard]], but we defer error checks
    auto cub_error_code = cub::DeviceScan::ExclusiveSum(
        nullptr, temp_storage_bytes, data, counts.size(), stream.get());
    // Allocate temporary storage
    CELER_DISCARD(cub_error_code);
    DeviceVector<char> temp_storage(temp_storage_bytes, stream_id);
    // Run exclusive prefix sum
    cub_error_code = cub::DeviceScan::ExclusiveSum(temp_storage.data(),
                                                   temp_storage_bytes,
                                                   data,
                                                   counts.size(),
                                                   stream.get());
    // Set the counter similar to the following
    // counters.num_secondaries = "last value in the counts object;
    CELER_DISCARD(cub_error_code);
    CELER_DEVICE_API_CALL(PeekAtLastError());

    // Copy the last element (accumulated total) back to host
    auto result
        = ItemCopier<size_type>{stream_id}(data.get() + counts.size() - 1);

    stream.sync();
    return result;
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
    size_type count,
    StreamId stream_id)
{
    CELER_EXPECT(count != 0);

    ScopedProfiling profile_this{"partition-initializers"};
#if CELER_USE_THRUST
    // Partition the indices based on the track initializer charge
    auto start = device_pointer_cast(init.indices.data());
    auto end = start + count;
    auto counters = device_pointer_cast(init.counters.data());
    auto cpucntrs = ItemCopier<CoreStateCounters>{stream_id}(counters.get());
    auto stencil = static_cast<TrackInitializer*>(init.initializers.data())
                   + cpucntrs.num_initializers - count;
    thrust::stable_partition(
        thrust_execute_on(stream_id),
        start,
        end,
        IsNeutralStencil{params.ptr<MemSpace::native>(), stencil});
    CELER_DEVICE_API_CALL(PeekAtLastError());
#else
    auto& stream = device().stream(stream_id);
    // CUB doesn't have a partition function that allows the user to specify
    // both an iterator for the values to use for selection and a function to
    // operate on that iterator. (This should change in the future.) So,
    // instead we create an iterator by using a functor to transform the
    // stencil values into boolean flags that determine how to partition
    // the indices.
    //
    // The initializers array is large. Use stencil to point to the start where
    // this array is being used
    auto counters = device_pointer_cast(init.counters.data());
    auto cpucntrs = ItemCopier<CoreStateCounters>{stream_id}(counters.get());
    auto stencil = static_cast<TrackInitializer*>(init.initializers.data())
                   + cpucntrs.num_initializers - count;
    DeviceVector<unsigned char> flags{count, stream_id};
#    if CELER_CUB_HAS_TRANSFORM || CELER_HIPCUB_HAS_TRANSFORM
    // HIP defines hipCUB functions as [[nodiscard]], but we defer error checks
    {
        auto cub_error_code = cub::DeviceTransform::Transform(
            stencil,
            flags.data(),
            count,
            IsNeutral{params.ptr<MemSpace::native>()},
            stream.get());
        CELER_DISCARD(cub_error_code);
    }
#    else
    thrust::transform(thrust_execute_on(stream_id),
                      stencil,
                      stencil + count,
                      flags.data(),
                      IsNeutral{params.ptr<MemSpace::native>()});
#    endif
    // Calling with nullptr causes the function to return the amount of working
    // space needed instead of invoking the kernel.
    size_t temp_storage_bytes = 0;
    // CUB doesn't support in-place partitioning, so use a counting iterator
    // because the indices are always sequential from zero
    auto start = thrust::make_counting_iterator<size_type>(0);
    auto data = device_pointer_cast(init.indices.data());
    auto cub_error_code
        = cub::DevicePartition::Flagged(nullptr,
                                        temp_storage_bytes,
                                        start,
                                        flags.data(),
                                        data,
                                        &(counters->num_neutral),
                                        count,
                                        stream.get());
    CELER_DISCARD(cub_error_code);
    // Allocate temporary storage
    DeviceVector<char> temp_storage(temp_storage_bytes, stream_id);
    // Partition the indices based on the track initializer charge
    cub_error_code = cub::DevicePartition::Flagged(temp_storage.data(),
                                                   temp_storage_bytes,
                                                   start,
                                                   flags.data(),
                                                   data,
                                                   &(counters->num_neutral),
                                                   count,
                                                   stream.get());
    CELER_DISCARD(cub_error_code);
    CELER_DEVICE_API_CALL(PeekAtLastError());
#endif
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
