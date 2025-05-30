//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/GridIdFinder.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/LdgIterator.hh"
#include "corecel/math/Algorithms.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Map an input grid to an ID type, returning invalid ID if outside bounds.
 *
 * The input grid should be a monotonic increasing series, and the
 * corresponding ID values should be one fewer (cell-centered data). Values
 * outside the grid bounds are unassigned, and grid points are attached to the
 * model ID above the corresponding value.
 *
 * \code
    GridIdFinder<MevEnergy, ActionId> find_model(energy, values);

    ActionId applicable_model = find_model(particle.energy());
   \endcode
 */
template<class KeyQuantity, class ValueId>
class GridIdFinder
{
    static_assert(KeyQuantity::unit_type::value() > 0, "Invalid Quantity");
    static_assert(sizeof(typename ValueId::size_type), "Invalid OpaqueId");

  public:
    //!@{
    //! \name Type aliases
    using argument_type = KeyQuantity;
    using result_type = ValueId;

    using SpanConstGrid = LdgSpan<typename KeyQuantity::value_type const>;
    using SpanConstValue = LdgSpan<result_type const>;
    //!@}

  public:
    // Construct from grid and values.
    inline CELER_FUNCTION GridIdFinder(SpanConstGrid, SpanConstValue);

    // Find the given grid point
    inline CELER_FUNCTION result_type operator()(argument_type arg) const;

  private:
    SpanConstGrid grid_;
    SpanConstValue value_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
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
    if (iter == grid_.end()
        || (iter == grid_.begin() && quant.value() != *iter))
    {
        // Higher than end point or below first point
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
}  // namespace celeritas
