//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/detail/TrackInitAlgorithms.cu
//---------------------------------------------------------------------------//
#include "TrackInitAlgorithms.hh"

#if CELERITAS_USE_CUDA
#    include <cub/device/device_select.cuh>
#    include <thrust/iterator/counting_iterator.h>
#    include <thrust/iterator/transform_iterator.h>
#elif CELERITAS_USE_HIP
#    include <hipcub/device/device_select.hpp>
#    if HIPCUB_VERSION >= 400100
#        include <hipcub/device/device_transform.hpp>
#        include <thrust/execution_policy.h>
#    else
#        include <thrust/transform.h>
#    endif
#    include <thrust/device_ptr.h>
#endif

#include "corecel/Macros.hh"
#include "corecel/data/DeviceVector.hh"
#include "corecel/data/ObserverPtr.device.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/Stream.hh"
#include "corecel/sys/Thrust.device.hh"

#if CELERITAS_USE_CUDA
using namespace cub;
#elif CELERITAS_USE_HIP
using namespace hipcub;
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

    // cub functions expect a cudaStream_t pointer for the stream
    using StreamT = CELER_DEVICE_API_SYMBOL(Stream_t);
    StreamT stream = device().stream(stream_id).get();

    // *** For testing with existing code for consistency
    DeviceVector<size_type> num_vacancies{1, stream_id};
    // *** For testing with existing code for consistency

    // Calling with nullptr causes the function to return the amount of working
    // space needed instead of invoking the kernel.
    size_t temp_storage_bytes = 0;
#if CELERITAS_USE_CUDA
    auto start = thrust::make_transform_iterator(
        thrust::make_counting_iterator<size_type>(0), TransformType{});
    auto flags = device_pointer_cast(status.data());
    auto result = device_pointer_cast(vacancies.data());
    DeviceSelect::FlaggedIf(nullptr,
                            temp_storage_bytes,
                            start,
                            flags,
                            result,
                            num_vacancies.data(),
                            vacancies.size(),
                            IsVacant{},
                            stream);

    // Allocate temporary storage
    DeviceAllocation temp_storage(temp_storage_bytes, stream_id);

    DeviceSelect::FlaggedIf(temp_storage.data(),
                            temp_storage_bytes,
                            start,
                            flags,
                            result,
                            num_vacancies.data(),
                            vacancies.size(),
                            IsVacant{},
                            stream);
#elif CELERITAS_USE_HIP
    auto start = device_pointer_cast(status.data());
    DeviceVector<unsigned char> flags{status.size(), stream_id};
    auto data = device_pointer_cast(vacancies.data());
#    if CELERITAS_USE_HIP && HIPCUB_VERSION >= 400100
    {
        auto cub_error_code = DeviceTransform::Transform(
            start, flags.data(), status.size(), IsVacant{}, stream);
        // HIP is particular about checking return codes from hipCUB functions,
        // so check for an error from the call and proceed accordingly
        if (cub_error_code)
            CELER_DEVICE_API_CALL(PeekAtLastError());
    }
#    else
    thrust::transform(thrust_execute_on(stream_id),
                      start,
                      start + count,
                      flags.data(),
                      IsVacant{});
#    endif
    auto cub_error_code = DeviceSelect::Flagged(nullptr,
                                                temp_storage_bytes,
                                                data,
                                                flags.data(),
                                                num_vacancies.data(),
                                                vacancies.size(),
                                                stream);

    // HIP is particular about checking return codes from hipCUB functions, so
    // check for an error from the call and proceed accordingly
    if (cub_error_code)
        CELER_DEVICE_API_CALL(PeekAtLastError());

    // Allocate temporary storage
    DeviceAllocation temp_storage(temp_storage_bytes, stream_id);

    cub_error_code = DeviceSelect::Flagged(temp_storage.data(),
                                           temp_storage_bytes,
                                           data,
                                           flags.data(),
                                           num_vacancies.data(),
                                           vacancies.size(),
                                           stream);

    // HIP is particular about checking return codes from hipCUB functions, so
    // check for an error from the call and proceed accordingly
    if (cub_error_code)
        CELER_DEVICE_API_CALL(PeekAtLastError());
#endif
    // *** For testing with existing code for consistency
    // *** Replace with the appropriate GPU counter
    auto num = ItemCopier<size_type>{stream_id}(num_vacancies.data());
    // *** For testing with existing code for consistency

    CELER_DEVICE_API_CALL(PeekAtLastError());
    // Number of vacancies
    return num;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
