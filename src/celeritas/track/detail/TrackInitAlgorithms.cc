//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/TrackInitAlgorithms.cc
//---------------------------------------------------------------------------//
#include "TrackInitAlgorithms.hh"

#include <algorithm>
#include <numeric>

#include "corecel/data/ObserverPtr.hh"
#include "corecel/math/Algorithms.hh"

using namespace celeritas::literals;

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Remove all elements in the vacancy vector that were flagged as active
 * tracks.
 */
void remove_if_alive(
    TrackInitStateData<Ownership::reference, MemSpace::host> const& init,
    StreamId)
{
    auto* start = init.vacancies.data().get();
    auto* counters = init.counters.data().get();
    auto* stop
        = std::remove_if(start, start + init.vacancies.size(), LogicalNot{});
    counters->num_vacancies = stop - start;
    return;
}

//---------------------------------------------------------------------------//
/*!
 * Do an exclusive scan of the number of secondaries produced by each track.
 *
 * For an input array x, this calculates the exclusive prefix sum y of the
 * array elements, i.e., \f$ y_i = \sum_{j=0}^{i-1} x_j \f$,
 * where \f$ y_0 = 0 \f$, and stores the result in the input array.
 *
 * The input size is one greater than the number of track slots so that the
 * final element will be the total accumulated value. The returned pointer
 * refers to that final element.
 */
ObserverPtr<size_type, MemSpace::host> exclusive_scan_counts(
    StateCollection<size_type, Ownership::reference, MemSpace::host> const&
        counts,
    StreamId)
{
    CELER_EXPECT(!counts.empty());
    auto* data = counts.data().get();
    auto* stop = data + counts.size();
#ifdef __cpp_lib_parallel_algorithm
    std::exclusive_scan(data, stop, data, 0_sz);
#else
    // Standard library shipped with GCC 8.5 does not include exclusive_scan
    // (I guess it's *too* exclusive)
    size_type acc = 0;
    for (; data != stop; ++data)
    {
        size_type current = *data;
        *data = acc;
        acc += current;
    }
#endif
    // Return last value (*not* past-the-end)
    return make_observer(stop - 1);
}

//---------------------------------------------------------------------------//
/*!
 * Sort the tracks that will be initialized in this step by charged/neutral.
 *
 * This partitions an array of indices used to access the track initializers
 * and the thread IDs of the initializers' parent tracks.
 */
void partition_initializers(
    CoreParams const& params,
    TrackInitStateData<Ownership::reference, MemSpace::host> const& init,
    size_type num_initializers,
    size_type count,
    StreamId)
{
    // Partition the indices based on the track initializer charge
    auto* start = init.indices.data().get();
    auto* end = start + count;
    auto* stencil = init.initializers.data().get() + num_initializers - count;
    std::stable_partition(
        start, end, IsNeutralStencil{params.ptr<MemSpace::native>(), stencil});
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
