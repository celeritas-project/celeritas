//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TrackInitAlgorithms.cc
//---------------------------------------------------------------------------//
#include "TrackInitAlgorithms.hh"

#include <algorithm>

#include "Utils.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Remove all elements in the vacancy vector that were flagged as active
 * tracks.
 */
template<>
size_type remove_if_alive<MemSpace::host>(Span<size_type> vacancies)
{
    auto end = std::remove_if(vacancies.data(),
                              vacancies.data() + vacancies.size(),
                              IsEqual{flag_id()});

    // New size of the vacancy vector
    size_type result = end - vacancies.data();
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Do an exclusive scan of the number of secondaries produced by each track.
 *
 * For an input array x, this calculates the exclusive prefix sum y of the
 * array elements, i.e., \f$ y_i = \sum_{j=0}^{i-1} x_j \f$,
 * where \f$ y_0 = 0 \f$, and stores the result in the input array.
 *
 * The return value is the sum of all elements in the input array.
 */
template<>
size_type exclusive_scan_counts<MemSpace::host>(Span<size_type> counts)
{
    // TODO: Use std::exclusive_scan when C++17 is adopted
    size_type acc = 0;
    for (auto& count_i : counts)
    {
        size_type current = count_i;
        count_i           = acc;
        acc += current;
    }
    return acc;
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
