//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/detail/TrackInitAlgorithms.cu
//---------------------------------------------------------------------------//
#include "TrackInitAlgorithms.hh"

#include <cub/cub.cuh>
// #include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "corecel/Macros.hh"
#include "corecel/data/ObserverPtr.device.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/Stream.hh"
#include "corecel/sys/Thrust.device.hh"

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

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    // *** For testing with existing code for consistency
    size_type* d_num_vacancies = nullptr;
    // *** Allocate temporary storage
    // *** For testing with existing code for consistency
    d_num_vacancies
        = static_cast<size_type*>(s.malloc_async(sizeof(size_type)));

    // The first call just computes the number of additional bytes needed for
    // the in-place selection. The nullptr value causes this instead of running
    // the function.
    auto start = thrust::make_transform_iterator(
        thrust::make_counting_iterator<size_type>(0), TransformType{});
    auto result = device_pointer_cast(vacancies.data());
    auto flags = device_pointer_cast(status.data());
    cub::DeviceSelect::FlaggedIf(d_temp_storage,
                                 temp_storage_bytes,
                                 start,
                                 flags,
                                 result,
                                 d_num_vacancies,
                                 vacancies.size(),
                                 IsVacant{},
                                 stream);

    // Allocate temporary storage
    d_temp_storage = s.malloc_async(temp_storage_bytes);

    cub::DeviceSelect::FlaggedIf(d_temp_storage,
                                 temp_storage_bytes,
                                 start,
                                 flags,
                                 result,
                                 d_num_vacancies,
                                 vacancies.size(),
                                 IsVacant{},
                                 stream);

    // Deallocate temporary storage
    s.free_async(d_temp_storage);

    // *** For testing with existing code for consistency
    // *** Replace with the appropriate GPU counter
    auto num = ItemCopier<size_type>{stream_id}(d_num_vacancies);
    s.free_async(d_num_vacancies);
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
