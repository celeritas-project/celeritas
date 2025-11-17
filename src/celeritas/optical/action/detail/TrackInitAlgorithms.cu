//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/detail/TrackInitAlgorithms.cu
//---------------------------------------------------------------------------//
#include "TrackInitAlgorithms.hh"

#if CELERITAS_USE_CUDA
#    include <cub/device/device_select.cuh>
#elif CELERITAS_HAVE_HIPCUB
#    include <hipcub/device/device_select.hpp>
#else
#    include <thrust/copy.h>
#    include <thrust/execution_policy.h>
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
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "corecel/Macros.hh"
#include "corecel/data/DeviceVector.hh"
#include "corecel/data/ObserverPtr.device.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/Stream.hh"
#include "corecel/sys/Thrust.device.hh"

#if CELERITAS_HAVE_HIPCUB
namespace cub = hipcub;
#endif

namespace celeritas
{
namespace optical
{
namespace detail
{

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
#ifdef CELER_USE_THRUST
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
    auto& stream = device().stream(stream_id);
    DeviceVector<size_type> num_vacancies{1, stream_id};
    auto start = thrust::make_transform_iterator(
        thrust::make_counting_iterator<size_type>(0), TransformType{});
#    if CELER_CUB_HAS_FLAGGEDIF
    // Calling with nullptr causes the function to return the amount of working
    // space needed instead of invoking the kernel.
    size_t temp_storage_bytes = 0;
    auto flags = device_pointer_cast(status.data());
    auto results = device_pointer_cast(vacancies.data());
    cub::DeviceSelect::FlaggedIf(nullptr,
                                 temp_storage_bytes,
                                 start,
                                 flags,
                                 results,
                                 num_vacancies.data(),
                                 vacancies.size(),
                                 IsVacant{},
                                 stream.get());
    // Allocate temporary storage
    DeviceVector<char> temp_storage(temp_storage_bytes, stream_id);
    cub::DeviceSelect::FlaggedIf(temp_storage.data(),
                                 temp_storage_bytes,
                                 start,
                                 flags,
                                 results,
                                 num_vacancies.data(),
                                 vacancies.size(),
                                 IsVacant{},
                                 stream.get());
#    else
    auto data = device_pointer_cast(status.data());
    DeviceVector<unsigned char> flags{status.size(), stream_id};
#        if CELER_CUB_HAS_TRANSFORM || CELER_HIPCUB_HAS_TRANSFORM
    // HIP defines hipCUB functions as [[nodiscard]], but we defer error checks
    {
        auto cub_error_code = cub::DeviceTransform::Transform(
            data, flags.data(), status.size(), IsVacant{}, stream.get());
        CELER_DISCARD(cub_error_code);
    }
#        else
    thrust::transform(thrust_execute_on(stream_id),
                      data,
                      data + status.size(),
                      flags.data(),
                      IsVacant{});
#        endif
    // Calling with nullptr causes the function to return the amount of working
    // space needed instead of invoking the kernel.
    size_t temp_storage_bytes = 0;
    auto results = device_pointer_cast(vacancies.data());
    auto cub_error_code = cub::DeviceSelect::Flagged(nullptr,
                                                     temp_storage_bytes,
                                                     start,
                                                     flags.data(),
                                                     results,
                                                     num_vacancies.data(),
                                                     vacancies.size(),
                                                     stream.get());
    CELER_DISCARD(cub_error_code);
    // Allocate temporary storage
    DeviceVector<char> temp_storage(temp_storage_bytes, stream_id);
    cub_error_code = cub::DeviceSelect::Flagged(temp_storage.data(),
                                                temp_storage_bytes,
                                                start,
                                                flags.data(),
                                                results,
                                                num_vacancies.data(),
                                                vacancies.size(),
                                                stream.get());
    CELER_DISCARD(cub_error_code);
#    endif
    CELER_DEVICE_API_CALL(PeekAtLastError());

    auto result = ItemCopier<size_type>{stream_id}(num_vacancies.data());

    stream.sync();
    return result;
#endif
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
