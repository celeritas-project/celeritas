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
#elif CELERITAS_HAVE_HIPCUB
#    include <hipcub/device/device_scan.hpp>
#    include <hipcub/device/device_select.hpp>
#else
#    include <thrust/execution_policy.h>
#    include <thrust/remove.h>
#    include <thrust/scan.h>

#    include "corecel/math/Algorithms.hh"  // For LogicalNot()
#endif
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

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

using namespace celeritas::literals;

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
    auto& stream = device().stream(stream_id);
#if CELER_USE_THRUST
    auto start = device_pointer_cast(init.vacancies.data());
    auto counters = device_pointer_cast(init.counters.data());
    auto end = thrust::remove_if(thrust_execute_on(stream_id),
                                 start,
                                 start + init.vacancies.size(),
                                 LogicalNot{});
    CELER_DEVICE_API_CALL(PeekAtLastError());

    // Update the number of vacancies
    size_type num_vacancies = end - start;
    Copier<size_type, MemSpace::device> copy{{&(counters->num_vacancies), 1},
                                             stream_id};
    copy(MemSpace::host, {&num_vacancies, 1});
    stream.sync();
#else
    // Calling with nullptr causes the function to return the amount of working
    // space needed instead of invoking the kernel.
    std::size_t temp_storage_bytes = 0;
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
#endif
    return;
}

//---------------------------------------------------------------------------//
/*!
 * Do an exclusive scan of the number of secondaries produced by each track.
 *
 * For an input array x, this calculates the exclusive prefix sum y of the
 * array elements, i.e., \f$ y_i = \sum_{j=0}^{i-1} x_j \f$,
 * where \f$ y_0 = 0 \f$, and stores the result in the input array.
 *
 * The returned pointer refers to the last element, which will hold the sum of
 * all elements in the input array once the stream's work completes.
 */
ObserverPtr<size_type, MemSpace::device> exclusive_scan_counts(
    StateCollection<size_type, Ownership::reference, MemSpace::device> const&
        counts,
    StreamId stream_id)
{
    ScopedProfiling profile_this{"prefix-sum-counts"};
    auto data = device_pointer_cast(counts.data());
#if CELER_USE_THRUST
    // Exclusive scan:
    thrust::exclusive_scan(
        thrust_execute_on(stream_id), data, data + counts.size(), data, 0_sz);
#else
    auto& stream = device().stream(stream_id);
    // Calling with nullptr causes the function to return the amount of working
    // space needed instead of invoking the kernel
    std::size_t temp_storage_bytes = 0;
    auto cub_error_code = cub::DeviceScan::ExclusiveSum(
        nullptr, temp_storage_bytes, data, counts.size(), stream.get());
    // HIP defines hipCUB functions as [[nodiscard]], but we defer error checks
    CELER_DISCARD(cub_error_code);
    // Allocate temporary storage
    DeviceVector<char> temp_storage(temp_storage_bytes, stream_id);
    // Run exclusive prefix sum
    cub_error_code = cub::DeviceScan::ExclusiveSum(temp_storage.data(),
                                                   temp_storage_bytes,
                                                   data,
                                                   counts.size(),
                                                   stream.get());
    CELER_DISCARD(cub_error_code);
#endif
    CELER_DEVICE_API_CALL(PeekAtLastError());
    // No synchronization since the next use of the results (data array), which
    // pulls the value from the results, will use another call on this stream
    return make_observer(counts.data().get() + counts.size() - 1);
}

//---------------------------------------------------------------------------//
/*!
 * Count the neutral tracks that will be initialized in this step.
 *
 * This calculates the inclusive prefix sum of \c IsNeutralNewTrack over all
 * track slots and stores it in the \c indices array: element \em i is the
 * number of neutral tracks among the first \em i + 1 initializers used in
 * this step.
 *
 * The number of new tracks is read from the device-side counters by the
 * flagging functor, and the scan size is fixed at the number of track slots,
 * so no synchronization with the host is needed.
 */
void scan_neutral_initializers(
    CoreParams const& params,
    TrackInitStateData<Ownership::reference, MemSpace::device> const& init,
    StreamId stream_id)
{
    CELER_EXPECT(!init.indices.empty());

    ScopedProfiling profile_this{"scan-neutral-initializers"};
    auto is_neutral = thrust::make_transform_iterator(
        thrust::make_counting_iterator<size_type>(0),
        IsNeutralNewTrack{params.ptr<MemSpace::native>(),
                          init.initializers.data().get(),
                          init.counters.data().get()});
    auto data = device_pointer_cast(init.indices.data());
#if CELER_USE_THRUST
    thrust::inclusive_scan(thrust_execute_on(stream_id),
                           is_neutral,
                           is_neutral + init.indices.size(),
                           data);
#else
    auto& stream = device().stream(stream_id);
    // Calling with nullptr causes the function to return the amount of working
    // space needed instead of invoking the kernel
    std::size_t temp_storage_bytes = 0;
    // HIP defines hipCUB functions as [[nodiscard]], but we defer error checks
    auto cub_error_code = cub::DeviceScan::InclusiveSum(nullptr,
                                                        temp_storage_bytes,
                                                        is_neutral,
                                                        data,
                                                        init.indices.size(),
                                                        stream.get());
    CELER_DISCARD(cub_error_code);
    // Allocate temporary storage
    DeviceVector<char> temp_storage(temp_storage_bytes, stream_id);
    // Run inclusive prefix sum
    cub_error_code = cub::DeviceScan::InclusiveSum(temp_storage.data(),
                                                   temp_storage_bytes,
                                                   is_neutral,
                                                   data,
                                                   init.indices.size(),
                                                   stream.get());
    CELER_DISCARD(cub_error_code);
#endif
    CELER_DEVICE_API_CALL(PeekAtLastError());
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
