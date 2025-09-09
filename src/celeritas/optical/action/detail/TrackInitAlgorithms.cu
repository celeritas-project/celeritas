//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/detail/TrackInitAlgorithms.cu
//---------------------------------------------------------------------------//
#include "TrackInitAlgorithms.hh"

#include <thrust/copy.h>
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
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
