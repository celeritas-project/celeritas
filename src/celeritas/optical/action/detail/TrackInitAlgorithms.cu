//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/detail/TrackInitAlgorithms.cu
//---------------------------------------------------------------------------//
#include "TrackInitAlgorithms.hh"

// CUDA has included cub since CUDA 11, but ROCm does not include hipCUB by
// default, so test for the availability of hipCUB and use thrust instead if
// it's unavailable. And some further checks for newer cub/hipCUB functions.
#define XSTR(x) STR(x)
#define STR(x) #x
#if CELER_USE_HIP
#    if CELERITAS_HAVE_HIPCUB
#        define CELER_USE_THRUST 0
#    else
#        define CELER_USE_THRUST 1
#    endif
#endif
#pragma message "The value of CELER_USE_HIP: " XSTR(CELER_USE_HIP)
#pragma message \
    "The value of CELERITAS_HAVE_HIPCUB: " XSTR(CELERITAS_HAVE_HIPCUB)
#pragma message "The value of CELER_USE_THRUST: " XSTR(CELER_USE_THRUST)
#if CELERITAS_USE_CUDA
#    include <cub/device/device_select.cuh>
#    include <cub/version.cuh>
#elif CELERITAS_USE_HIP && CELERITAS_HAVE_HIPCUB
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
#    include <thrust/copy.h>
#    include <thrust/iterator/counting_iterator.h>
#    include <thrust/iterator/transform_iterator.h>
#endif

#include "corecel/Macros.hh"
#include "corecel/data/DeviceVector.hh"
#include "corecel/data/ObserverPtr.device.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/Stream.hh"
#include "corecel/sys/Thrust.device.hh"

#if CELERITAS_USE_HIP && CELERITAS_HAVE_HIPCUB
namespace cub = hipcub;
#endif

namespace celeritas
{
namespace optical
{
namespace detail
{
#if CELER_USE_THRUST
namespace
{
//---------------------------------------------------------------------------//
struct TransformType
{
    CELER_FUNCTION TrackSlotId operator()(size_type i) const
    {
        return TrackSlotId{i};
    }
};
//---------------------------------------------------------------------------//
}  // namespace
#endif

//---------------------------------------------------------------------------//
/*!
 * Compact the \c TrackSlotIds of the inactive tracks.
 *
 * \return Number of vacant track slots
 */
size_type copy_if_vacant(TrackStatusRef<MemSpace::device> const& status,
                         TrackSlotRef<MemSpace::device> const& vacancies,
                         StreamId stream_id)
{
    CELER_EXPECT(status.size() == vacancies.size());

    ScopedProfiling profile_this{"copy-if-vacant"};
#if CELER_USE_THRUST
    auto start = thrust::make_transform_iterator(
        thrust::make_counting_iterator<size_type>(0), TransformType{});
    auto result = device_pointer_cast(vacancies.data());
    auto end = thrust::copy_if(thrust_execute_on(stream_id),
                               start,
                               start + vacancies.size(),
                               device_pointer_cast(status.data()),
                               result,
                               IsVacant{});
    CELER_DEVICE_API_CALL(PeekAtLastError());

    return end - result;
#else
    // cub functions expect a cudaStream_t pointer for the stream
    using StreamT = CELER_DEVICE_API_SYMBOL(Stream_t);
    StreamT stream = device().stream(stream_id).get();

    DeviceVector<size_type> num_vacancies{1, stream_id};

    auto start = device_pointer_cast(status.data());
    DeviceVector<unsigned char> flags{status.size(), stream_id};
#    if CELER_CUB_HAS_TRANSFORM || CELER_HIPCUB_HAS_TRANSFORM
    // HIP defines hipCUB functions as [[nodiscard]], but we defer error checks
    {
        auto cub_error_code = cub::DeviceTransform::Transform(
            start, flags.data(), status.size(), IsVacant{}, stream);
    }
#    else
    thrust::transform(thrust_execute_on(stream_id),
                      start,
                      start + status.size(),
                      flags.data(),
                      IsVacant{});
#    endif
    // Calling with nullptr causes the function to return the amount of working
    // space needed instead of invoking the kernel.
    size_t temp_storage_bytes = 0;
    auto data = device_pointer_cast(vacancies.data());
    auto cub_error_code = cub::DeviceSelect::Flagged(nullptr,
                                                     temp_storage_bytes,
                                                     data,
                                                     flags.data(),
                                                     num_vacancies.data(),
                                                     vacancies.size(),
                                                     stream);
    // Allocate temporary storage
    DeviceAllocation temp_storage(temp_storage_bytes, stream_id);
    cub_error_code = cub::DeviceSelect::Flagged(temp_storage.data(),
                                                temp_storage_bytes,
                                                data,
                                                flags.data(),
                                                num_vacancies.data(),
                                                vacancies.size(),
                                                stream);

    auto num = ItemCopier<size_type>{stream_id}(num_vacancies.data());
    CELER_DEVICE_API_CALL(PeekAtLastError());
    // Number of vacancies
    return num;
#endif
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
