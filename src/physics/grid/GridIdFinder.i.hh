//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GridIdFinder.i.hh
//---------------------------------------------------------------------------//
#include "base/Algorithms.hh"
#include "base/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from grid and values.
 *
 * \todo Construct from reference to collections and ranges so we can benefit
 * from
 * __ldg as needed
 */
template<class K, class V>
CELER_FUNCTION
GridIdFinder<K, V>::GridIdFinder(SpanConstGrid grid, SpanConstValue value)
    : grid_(grid), value_(value)
{
    CELER_EXPECT(grid_.size() == value_.size() + 1);
}

//---------------------------------------------------------------------------//
/*!
 * Find the ID corresponding to the given value.
 */
template<class K, class V>
CELER_FUNCTION auto GridIdFinder<K, V>::operator()(argument_type quant) const
    -> result_type
{
    auto iter
        = celeritas::lower_bound(grid_.begin(), grid_.end(), quant.value());
    if (iter == grid_.end())
    {
        // Higher than end point
        return {};
    }
    else if (iter == grid_.begin() && quant.value() != *iter)
    {
        // Below first point
        return {};
    }
    else if (iter + 1 == grid_.end() || quant.value() != *iter)
    {
        // Exactly on end grid point, or not on a grid point at all: move to
        // previous bin
        --iter;
    }
    return value_[iter - grid_.begin()];
}

//---------------------------------------------------------------------------//
} // namespace celeritas
