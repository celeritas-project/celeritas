//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/DeviceAlgorithms.device.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/DeviceRuntimeApi.hh"

#if CELERITAS_USE_CUDA
#    include <cub/device/device_reduce.cuh>
#    include <cub/device/device_scan.cuh>
#    include <cub/device/device_select.cuh>
#elif CELERITAS_HAVE_HIPCUB
#    include <hipcub/device/device_reduce.hpp>
#    include <hipcub/device/device_scan.hpp>
#    include <hipcub/device/device_select.hpp>
#endif
#if CELER_USE_THRUST
#    include <thrust/copy.h>
#    include <thrust/device_ptr.h>
#    include <thrust/reduce.h>
#    include <thrust/remove.h>
#    include <thrust/scan.h>
#endif

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/Stream.hh"
#include "corecel/sys/Thrust.device.hh"

#include "Copier.hh"
#include "DeviceVector.hh"
#include "ObserverPtr.device.hh"
#include "ObserverPtr.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//! Negate a unary predicate (for compaction with Thrust)
template<class Pred>
struct NotPredicate
{
    Pred pred;

    template<class T>
    CELER_FUNCTION bool operator()(T const& value) const
    {
        return !pred(value);
    }
};
}  // namespace detail

//---------------------------------------------------------------------------//
/*!
 * Stably compact, in place, the items that satisfy a predicate.
 *
 * The number of selected items is written to \c num_selected in device
 * memory, so this does not synchronize the stream unless the CUB selection
 * algorithm is unavailable (\c CELER_USE_THRUST). The items past the selected
 * ones are left in an unspecified state.
 */
template<class T, class Pred>
void select_if_async(ObserverPtr<T, MemSpace::device> data,
                     size_type num_items,
                     ObserverPtr<size_type, MemSpace::device> num_selected,
                     Pred pred,
                     StreamId stream_id)
{
    CELER_EXPECT(data || num_items == 0);
    CELER_EXPECT(num_selected);

#if CELER_USE_THRUST
    auto start = device_pointer_cast(data);
    auto stop = thrust::remove_if(thrust_execute_on(stream_id),
                                  start,
                                  start + num_items,
                                  detail::NotPredicate<Pred>{pred});
    CELER_DEVICE_API_CALL(PeekAtLastError());

    // Store the number of selected items on device (synchronizes)
    size_type host_num_selected = stop - start;
    Copier<size_type, MemSpace::device> copy{{num_selected.get(), 1},
                                             stream_id};
    copy(MemSpace::host, {&host_num_selected, 1});
    device().stream(stream_id).sync();
#else
#    if CELERITAS_HAVE_HIPCUB
    namespace cub = hipcub;
#    endif
    auto& stream = device().stream(stream_id);
    // Calling with nullptr returns the amount of working space needed
    std::size_t temp_storage_bytes = 0;
    // HIP defines hipCUB functions as [[nodiscard]], but we defer error checks
    auto cub_error_code = cub::DeviceSelect::If(nullptr,
                                                temp_storage_bytes,
                                                data.get(),
                                                num_selected.get(),
                                                num_items,
                                                pred,
                                                stream.get());
    CELER_DISCARD(cub_error_code);
    DeviceVector<char> temp_storage(temp_storage_bytes, stream_id);
    cub_error_code = cub::DeviceSelect::If(temp_storage.data(),
                                           temp_storage_bytes,
                                           data.get(),
                                           num_selected.get(),
                                           num_items,
                                           pred,
                                           stream.get());
    CELER_DISCARD(cub_error_code);
    CELER_DEVICE_API_CALL(PeekAtLastError());
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Stably copy the input items that satisfy a predicate.
 *
 * The input may be any random-access iterator. The number of selected items
 * is written to \c num_selected in device memory, so this does not
 * synchronize the stream unless the CUB selection algorithm is unavailable
 * (\c CELER_USE_THRUST).
 */
template<class InputIt, class T, class Pred>
void copy_if_async(InputIt input,
                   ObserverPtr<T, MemSpace::device> output,
                   size_type num_items,
                   ObserverPtr<size_type, MemSpace::device> num_selected,
                   Pred pred,
                   StreamId stream_id)
{
    CELER_EXPECT(output || num_items == 0);
    CELER_EXPECT(num_selected);

#if CELER_USE_THRUST
    auto start = device_pointer_cast(output);
    auto stop = thrust::copy_if(
        thrust_execute_on(stream_id), input, input + num_items, start, pred);
    CELER_DEVICE_API_CALL(PeekAtLastError());

    // Store the number of selected items on device (synchronizes)
    size_type host_num_selected = stop - start;
    Copier<size_type, MemSpace::device> copy{{num_selected.get(), 1},
                                             stream_id};
    copy(MemSpace::host, {&host_num_selected, 1});
    device().stream(stream_id).sync();
#else
#    if CELERITAS_HAVE_HIPCUB
    namespace cub = hipcub;
#    endif
    auto& stream = device().stream(stream_id);
    std::size_t temp_storage_bytes = 0;
    auto cub_error_code = cub::DeviceSelect::If(nullptr,
                                                temp_storage_bytes,
                                                input,
                                                output.get(),
                                                num_selected.get(),
                                                num_items,
                                                pred,
                                                stream.get());
    CELER_DISCARD(cub_error_code);
    DeviceVector<char> temp_storage(temp_storage_bytes, stream_id);
    cub_error_code = cub::DeviceSelect::If(temp_storage.data(),
                                           temp_storage_bytes,
                                           input,
                                           output.get(),
                                           num_selected.get(),
                                           num_items,
                                           pred,
                                           stream.get());
    CELER_DISCARD(cub_error_code);
    CELER_DEVICE_API_CALL(PeekAtLastError());
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the sum of a fixed number of input values.
 *
 * The result is written to \c result in device memory, so this does not
 * synchronize the stream unless the CUB reduction is unavailable (HIP without
 * hipCUB).
 */
template<class InputIt, class T>
void reduce_sum_async(InputIt input,
                      ObserverPtr<T, MemSpace::device> result,
                      size_type num_items,
                      StreamId stream_id)
{
    CELER_EXPECT(result);

#if CELERITAS_USE_CUDA || CELERITAS_HAVE_HIPCUB
#    if CELERITAS_HAVE_HIPCUB
    namespace cub = hipcub;
#    endif
    auto& stream = device().stream(stream_id);
    std::size_t temp_storage_bytes = 0;
    auto cub_error_code = cub::DeviceReduce::Sum(nullptr,
                                                 temp_storage_bytes,
                                                 input,
                                                 result.get(),
                                                 num_items,
                                                 stream.get());
    CELER_DISCARD(cub_error_code);
    DeviceVector<char> temp_storage(temp_storage_bytes, stream_id);
    cub_error_code = cub::DeviceReduce::Sum(temp_storage.data(),
                                            temp_storage_bytes,
                                            input,
                                            result.get(),
                                            num_items,
                                            stream.get());
    CELER_DISCARD(cub_error_code);
    CELER_DEVICE_API_CALL(PeekAtLastError());
#else
    // Store the sum on device (synchronizes)
    T host_result = thrust::reduce(
        thrust_execute_on(stream_id), input, input + num_items, T{0});
    CELER_DEVICE_API_CALL(PeekAtLastError());
    Copier<T, MemSpace::device> copy{{result.get(), 1}, stream_id};
    copy(MemSpace::host, {&host_result, 1});
    device().stream(stream_id).sync();
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the inclusive prefix sum of a fixed number of input values.
 *
 * The input may be any random-access iterator (e.g., a transform of a
 * counting iterator whose functor reads a device-side size). This does not
 * synchronize the stream.
 */
template<class InputIt, class T>
void inclusive_sum_async(InputIt input,
                         ObserverPtr<T, MemSpace::device> output,
                         size_type num_items,
                         StreamId stream_id)
{
    CELER_EXPECT(output || num_items == 0);

#if CELER_USE_THRUST
    thrust::inclusive_scan(thrust_execute_on(stream_id),
                           input,
                           input + num_items,
                           device_pointer_cast(output));
    CELER_DEVICE_API_CALL(PeekAtLastError());
#else
#    if CELERITAS_HAVE_HIPCUB
    namespace cub = hipcub;
#    endif
    auto& stream = device().stream(stream_id);
    std::size_t temp_storage_bytes = 0;
    auto cub_error_code = cub::DeviceScan::InclusiveSum(nullptr,
                                                        temp_storage_bytes,
                                                        input,
                                                        output.get(),
                                                        num_items,
                                                        stream.get());
    CELER_DISCARD(cub_error_code);
    DeviceVector<char> temp_storage(temp_storage_bytes, stream_id);
    cub_error_code = cub::DeviceScan::InclusiveSum(temp_storage.data(),
                                                   temp_storage_bytes,
                                                   input,
                                                   output.get(),
                                                   num_items,
                                                   stream.get());
    CELER_DISCARD(cub_error_code);
    CELER_DEVICE_API_CALL(PeekAtLastError());
#endif
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
