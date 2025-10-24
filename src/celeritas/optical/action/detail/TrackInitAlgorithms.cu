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

    using StreamT = CELER_DEVICE_API_SYMBOL(Stream_t);
    Stream& s = device().stream(stream_id);
    StreamT stream = s.get();

    // The first call just computes the number of additional bytes needed for
    // the in-place selection. Calling with nullptr causes this instead of
    // invoking the kernel.
    size_t temp_storage_bytes = 0;
    // *** For testing with existing code for consistency
    DeviceVector<size_type> num_vacancies{1, stream_id};
    // *** For testing with existing code for consistency

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
    void* d_temp_storage = s.malloc_async(temp_storage_bytes);

    DeviceSelect::FlaggedIf(d_temp_storage,
                            temp_storage_bytes,
                            start,
                            flags,
                            result,
                            num_vacancies.data(),
                            vacancies.size(),
                            IsVacant{},
                            stream);

    // Deallocate temporary storage
    s.free_async(d_temp_storage);

    // *** For testing with existing code for consistency
    // *** Replace with the appropriate GPU counter
    auto num = ItemCopier<size_type>{stream_id}(num_vacancies.data());
    // *** For testing with existing code for consistency

    CELER_DEVICE_API_CALL(PeekAtLastError());
    // Number of vacancies
    return num;
    // auto start = thrust::make_transform_iterator(
    // thrust::make_counting_iterator<size_type>(0), TransformType{});
    // auto result = device_pointer_cast(vacancies.data());
    // auto end = thrust::copy_if(thrust_execute_on(stream_id),
    // start,
    // start + vacancies.size(),
    // device_pointer_cast(status.data()),
    // result,
    // IsVacant{});
    // CELER_DEVICE_API_CALL(PeekAtLastError());

    // return end - result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
