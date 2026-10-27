//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/detail/TrackInitAlgorithms.cu
//---------------------------------------------------------------------------//
#include "TrackInitAlgorithms.hh"

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "corecel/data/DeviceAlgorithms.device.hh"
#include "corecel/data/ObserverPtr.hh"
#include "corecel/sys/ScopedProfiling.hh"

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
//! Whether the given track slot is vacant
struct IsVacantAt
{
    TrackStatus const* status;

    CELER_FUNCTION bool operator()(TrackSlotId slot) const
    {
        return IsVacant{}(status[slot.unchecked_get()]);
    }
};
//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Compact the \c TrackSlotIds of the inactive tracks.
 */
void copy_if_vacant(TrackStatusRef<MemSpace::device> const& status,
                    TrackInitRef<MemSpace::device> const& init,
                    StreamId stream_id)
{
    CELER_EXPECT(status.size() == init.vacancies.size());

    ScopedProfiling profile_this{"copy-if-vacant"};
    auto* counters = init.counters.data().get();
    copy_if_async(
        thrust::make_transform_iterator(
            thrust::make_counting_iterator<size_type>(0), TransformType{}),
        init.vacancies.data(),
        init.vacancies.size(),
        ObserverPtr<size_type, MemSpace::device>{&counters->num_vacancies},
        IsVacantAt{status.data().get()},
        stream_id);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
