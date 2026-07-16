//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/OffloadAlgorithms.cu
//---------------------------------------------------------------------------//
#include "OffloadAlgorithms.hh"

#include <thrust/device_ptr.h>
#include <thrust/functional.h>
// TODO: Move these two headers inside the CELER_USE_THRUST block once the
//       remove_if function is ported to CUB/hipCUB
#include <thrust/execution_policy.h>
#include <thrust/remove.h>
#if CELER_CUB_HAS_TRANSFORM_REDUCE
#    include <cub/device/device_reduce.cuh>
#elif CELER_HIPCUB_HAS_TRANSFORM_REDUCE
#    include <hipcub/device/device_reduce.hpp>
#else
#    include <thrust/transform_reduce.h>
#endif
#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/data/Copier.hh"
#include "corecel/data/DeviceVector.hh"
#include "corecel/data/ObserverPtr.device.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/Stream.hh"
#include "corecel/sys/Thrust.device.hh"
#if CELER_USE_THRUST
#    include "celeritas/optical/TrackExecutor.hh"
#    include "celeritas/optical/action/ActionLauncher.device.hh"

#    include "UpdatePendingExecutor.hh"
#endif

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
    auto& stream = device().stream(stream_id);
    auto start = thrust::device_pointer_cast(buffer.data().get());
#if CELER_CUB_HAS_TRANSFORM_REDUCE || CELER_HIPCUB_HAS_TRANSFORM_REDUCE
    size_t temp_storage_bytes = 0;
    auto counters = device_pointer_cast(state.ref().init.counters.data());
    // Can't pass the current value of the num_pending counter on device to
    // the reduction, but need it to initialize the sum
    auto cpucntrs = ItemCopier<CoreStateCounters>{stream_id}(counters.get());
    // HIP defines hipCUB functions as [[nodiscard]], but we defer error checks
    // Calling with nullptr causes the function to return the amount of working
    // space needed instead of invoking the kernel
    auto cub_error_code = cub::DeviceReduce::TransformReduce(
        nullptr,
        temp_storage_bytes,
        start + offset,
        &(counters->num_pending),
        size - offset,
        thrust::plus<size_type>(),
        celeritas::optical::GetNumPhotons<GeneratorDistributionData>{},
        cpucntrs.num_pending,
        stream.get());
    CELER_DISCARD(cub_error_code);
    DeviceVector<char> temp_storage(temp_storage_bytes, stream_id);
    // Run reduction
    cub_error_code = cub::DeviceReduce::TransformReduce(
        temp_storage.data(),
        temp_storage_bytes,
        start + offset,
        &(counters->num_pending),
        size - offset,
        thrust::plus<size_type>(),
        celeritas::optical::GetNumPhotons<GeneratorDistributionData>{},
        cpucntrs.num_pending,
        stream.get());
    CELER_DISCARD(cub_error_code);
    CELER_DEVICE_API_CALL(PeekAtLastError());
#else
    size_type count = thrust::transform_reduce(
        thrust_execute_on(stream),
        start + offset,
        start + size,
        celeritas::optical::GetNumPhotons<GeneratorDistributionData>{},
        0_sz,
        thrust::plus<size_type>());
    CELER_DEVICE_API_CALL(PeekAtLastError());
    // Update the number of pending optical photons
    auto execute_thread = make_single_track_executor(
        params->ptr<MemSpace::native>(),
        state.ptr(),
        optical::detail::UpdatePendingExecutor{count});
    static KernelLauncher<decltype(execute_thread)> const launch_kernel(
        "update-pending");
    launch_kernel(1, state.stream_id(), execute_thread);
#endif
    stream.sync();
    return;
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template size_type
remove_if_invalid(ItemsRef<GeneratorDistributionData, MemSpace::device> const&,
                  size_type,
                  size_type,
                  StreamId);
template size_type
remove_if_invalid(ItemsRef<WlsDistributionData, MemSpace::device> const&,
                  size_type,
                  size_type,
                  StreamId);

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
