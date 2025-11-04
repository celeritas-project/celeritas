//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/detail/TrackInitAlgorithms.cu
//---------------------------------------------------------------------------//
#include "TrackInitAlgorithms.hh"

#if CELERITAS_USE_CUDA
#    include <cub/device/device_select.cuh>
#elif CELERITAS_USE_HIP
#    include <hipcub/device/device_select.hpp>
#endif
// #include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

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

    {
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
    }

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
