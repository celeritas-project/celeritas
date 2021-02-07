//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GridIdFinder.i.hh
//---------------------------------------------------------------------------//

#include "base/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from grid and values.
 *
 * \todo maybe construct from pies/slices?
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
CELER_FUNCTION auto GridIdFinder<K, V>::operator()(argument_type) const
    -> result_type
{
    CELER_NOT_IMPLEMENTED("GridIdFinder");
}

//---------------------------------------------------------------------------//
} // namespace celeritas
