//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/TrackInitAlgorithms.cc
//---------------------------------------------------------------------------//
#include "TrackInitAlgorithms.hh"

#include <algorithm>
#include <numeric>

#include "Utils.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Remove all elements in the vacancy vector that were flagged as active
 * tracks.
 *
 * \return New size of the vacancy vector
 */
size_type remove_if_alive(
    StateCollection<TrackSlotId, Ownership::reference, MemSpace::host> const&
        vacancies,
    StreamId)
{
    auto* start = static_cast<TrackSlotId*>(vacancies.data());
    auto* stop
        = std::remove_if(start, start + vacancies.size(), IsEqual{occupied()});
    return stop - start;
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
 * final element will be the total accumulated value.
 */
size_type exclusive_scan_counts(
    StateCollection<size_type, Ownership::reference, MemSpace::host> const& counts,
    StreamId)
{
    CELER_EXPECT(!counts.empty());
    auto* data = static_cast<size_type*>(counts.data());
#ifdef __cpp_lib_parallel_algorithm
    auto* stop
        = std::exclusive_scan(data, data + counts.size(), data, size_type{0});
#else
    // Standard library shipped with GCC 8.5 does not include exclusive_scan
    // (I guess it's *too* exclusive)
    size_type acc = 0;
    auto* const stop = data + counts.size();
    for (; data != stop; ++data)
    {
        size_type current = *data;
        *data = acc;
        acc += current;
    }
#endif
    // Return the final value
    return *(stop - 1);
}

//---------------------------------------------------------------------------//
/*!
 * Sort the tracks that will be initialized in this step by charged/neutral.
 *
 * The return value is the partition point between neutral and charged tracks.
 */
size_type partition_initializers(
    CoreParams const& params,
    Collection<TrackInitializer, Ownership::reference, MemSpace::host> const& init,
    CoreStateCounters const& counters,
    size_type count,
    StreamId)
{
    auto* end = static_cast<TrackInitializer*>(init.data())
                + counters.num_initializers;
    auto* start = end - count;
    auto* partition_index = std::stable_partition(
        start, end, IsNeutral{params.ptr<MemSpace::native>()});

    // Number of neutral tracks
    return partition_index - start;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
