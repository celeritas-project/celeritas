//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/OffloadAlgorithms.cu
//---------------------------------------------------------------------------//
#include "OffloadAlgorithms.hh"

#include <type_traits>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
// TODO: Move these two headers inside the #else block once the
//       remove_if_invalid function is ported to CUB/hipCUB
#include <thrust/execution_policy.h>
#include <thrust/remove.h>

#include "corecel/data/ObserverPtr.hh"
#if CELERITAS_USE_CUDA
#    include <cub/device/device_reduce.cuh>
#    include <thrust/iterator/transform_iterator.h>
#elif CELERITAS_USE_HIP && CELERITAS_HAVE_HIPCUB
#    include <hipcub/device/device_reduce.hpp>
#    include <thrust/iterator/transform_iterator.h>
#else
#    include <thrust/transform_reduce.h>
#endif
#include "corecel/Assert.hh"
#include "corecel/data/Copier.hh"
#include "corecel/data/DeviceVector.hh"
#include "corecel/data/ObserverPtr.device.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/Stream.hh"
#include "corecel/sys/Thrust.device.hh"
#include "celeritas/optical/TrackExecutor.hh"
#include "celeritas/optical/action/ActionLauncher.device.hh"

#include "UpdatePendingExecutor.hh"

#if CELERITAS_HAVE_HIPCUB
namespace cub = hipcub;
#endif

using namespace celeritas::literals;

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Remove all invalid distributions from the buffer.
 *
 * \return Total number of valid distributions in the buffer
 */
template<class T>
size_type remove_if_invalid(ItemsRef<T, MemSpace::device> const& buffer,
                            size_type offset,
                            size_type size,
                            StreamId stream_id)
{
    ScopedProfiling profile_this{"remove-if-invalid"};
    auto start = thrust::device_pointer_cast(buffer.data().get());
    auto stop = thrust::remove_if(thrust_execute_on(stream_id),
                                  start + offset,
                                  start + size,
                                  LogicalNot{});
    CELER_DEVICE_API_CALL(PeekAtLastError());
    return stop - start;
}

//---------------------------------------------------------------------------//
/*!
 * Count the number of optical photons in the distributions and add these to
 * the number of pending tracks.
 */
void count_num_photons(
    SPConstOpticalParams params,
    optical::CoreState<MemSpace::device>& state,
    ItemsRef<GeneratorDistributionData, MemSpace::device> const& buffer,
    size_type offset,
    size_type size,
    StreamId stream_id)
{
    ScopedProfiling profile_this{"count-num-photons"};
    CELER_EXPECT(params);
    auto& stream = device().stream(stream_id);
    auto start = device_pointer_cast(buffer.data());
#if CELERITAS_USE_CUDA || (CELERITAS_USE_HIP && CELERITAS_HAVE_HIPCUB)
    size_t temp_storage_bytes = 0;
    // This could be allocated once and reused for each call
    DeviceVector<size_type> result(1, stream_id);
    auto transform = thrust::transform_iterator(
        start + offset,
        celeritas::optical::GetNumPhotons<GeneratorDistributionData>());
    // Calling with nullptr causes the function to return the amount of working
    // space needed instead of invoking the kernel
    // Note: The CUB/hipCUB functions need the number of entries being
    // processed instead of the end of the entries, so we need to pass the end
    // of the distributions (size) minus the starting point, which is offset
    // Compute summation
    auto cub_error_code = cub::DeviceReduce::Sum(nullptr,
                                                 temp_storage_bytes,
                                                 transform,
                                                 result.data(),
                                                 size - offset,
                                                 stream.get());
    // HIP defines hipCUB functions as [[nodiscard]], but we defer error checks
    CELER_DISCARD(cub_error_code);
    // Note: Reductions are done in place but allocate 1 byte, so this could
    // be done once and reused
    DeviceVector<char> temp_storage(temp_storage_bytes, stream_id);
    cub_error_code = cub::DeviceReduce::Sum(temp_storage.data(),
                                            temp_storage_bytes,
                                            transform,
                                            result.data(),
                                            size - offset,
                                            stream.get());
    size_type* count{
        result.data()};  // Must match variable name in #else below for thrust
    CELER_DISCARD(cub_error_code);
#else
    size_type count = thrust::transform_reduce(
        thrust_execute_on(stream_id),
        start + offset,
        start + size,
        celeritas::optical::GetNumPhotons<GeneratorDistributionData>{},
        0_sz,
        thrust::plus<size_type>());
    // If there aren't any new photons, skip updating the counter. Can't do the
    // same check with the CUB/hipCUB functions because the counter is device
    // resident.
    if (count == 0)
    {
        CELER_DEVICE_API_CALL(PeekAtLastError());
        return;
    }
#endif
    CELER_DEVICE_API_CALL(PeekAtLastError());
    // Update the number of pending optical photons
    optical::detail::UpdatePendingExecutor<decltype(count)> execute_thread{
        state.ref().init.counters.data(), count};
    static KernelLauncher<decltype(execute_thread)> const launch_kernel(
        "update-pending");
    launch_kernel(1, stream_id, execute_thread);
    return;
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template size_type remove_if_invalid(
    ItemsRef<GeneratorDistributionData, MemSpace::device> const&,
    size_type,
    size_type,
    StreamId);
template size_type remove_if_invalid(
    ItemsRef<WlsDistributionData, MemSpace::device> const&,
    size_type,
    size_type,
    StreamId);

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
