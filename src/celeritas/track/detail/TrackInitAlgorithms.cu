//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/TrackInitAlgorithms.cu
//---------------------------------------------------------------------------//
#include "TrackInitAlgorithms.hh"

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "corecel/data/DeviceAlgorithms.device.hh"
#include "corecel/data/ObserverPtr.hh"
#include "corecel/sys/ScopedProfiling.hh"

#include "../Utils.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Whether the track slot is being used
 */
struct NotNull
{
    CELER_FUNCTION bool operator()(TrackSlotId a) const noexcept
    {
        return static_cast<bool>(a);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Remove all elements in the vacancy vector that were flagged as active
 * tracks.
 */
void remove_if_alive(
    TrackInitStateData<Ownership::reference, MemSpace::device> const& init,
    StreamId stream_id)
{
    ScopedProfiling profile_this{"remove-if-alive"};
    auto* counters = init.counters.data().get();
    select_if_async(
        init.vacancies.data(),
        init.vacancies.size(),
        ObserverPtr<size_type, MemSpace::device>{&counters->num_vacancies},
        NotNull{},
        stream_id);
}

//---------------------------------------------------------------------------//
/*!
 * Do an exclusive scan of the number of secondaries produced by each track.
 *
 * For an input array x, this calculates the exclusive prefix sum y of the
 * array elements, i.e., \f$ y_i = \sum_{j=0}^{i-1} x_j \f$,
 * where \f$ y_0 = 0 \f$, and stores the result in the input array.
 *
 * The returned pointer refers to the last element, which will hold the sum of
 * all elements in the input array once the stream's work completes.
 */
ObserverPtr<size_type, MemSpace::device> exclusive_scan_counts(
    StateCollection<size_type, Ownership::reference, MemSpace::device> const&
        counts,
    StreamId stream_id)
{
    ScopedProfiling profile_this{"prefix-sum-counts"};
    // Scan in place
    exclusive_sum_async(
        counts.data().get(), counts.data(), counts.size(), stream_id);
    // No synchronization since the next use of the results (data array), which
    // pulls the value from the results, will use another call on this stream
    return make_observer(counts.data().get() + counts.size() - 1);
}

//---------------------------------------------------------------------------//
/*!
 * Count the neutral tracks that will be initialized in this step.
 *
 * This calculates the inclusive prefix sum of \c IsNeutralNewTrack over all
 * track slots and stores it in the \c indices array: element \em i is the
 * number of neutral tracks among the first \em i + 1 initializers used in
 * this step.
 *
 * The number of new tracks is read from the device-side counters by the
 * flagging functor, and the scan size is fixed at the number of track slots,
 * so no synchronization with the host is needed.
 */
void scan_neutral_initializers(
    CoreParams const& params,
    TrackInitStateData<Ownership::reference, MemSpace::device> const& init,
    StreamId stream_id)
{
    CELER_EXPECT(!init.indices.empty());

    ScopedProfiling profile_this{"scan-neutral-initializers"};
    auto is_neutral = thrust::make_transform_iterator(
        thrust::make_counting_iterator<size_type>(0),
        IsNeutralNewTrack{params.ptr<MemSpace::native>(),
                          init.initializers.data().get(),
                          init.counters.data().get()});
    inclusive_sum_async(
        is_neutral, init.indices.data(), init.indices.size(), stream_id);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
