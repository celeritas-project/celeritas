//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GridIdFinder.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Span.hh"

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
    GridIdFinder<MevEnergy, ModelId> find_model(energy, values);

    ModelId applicable_model = find_model(particle.energy());
   \endcode
 */
template<class KeyQuantity, class ValueId>
class GridIdFinder
{
    static_assert(KeyQuantity::unit_type::value() > 0, "Invalid Quantity");
    static_assert(sizeof(typename ValueId::value_type), "Invalid OpaqueId");

  public:
    //!@{
    //! Type aliases
    using argument_type = KeyQuantity;
    using result_type   = ValueId;

    using SpanConstGrid  = Span<const typename KeyQuantity::value_type>;
    using SpanConstValue = Span<const result_type>;
    //!@}

  public:
    // Construct from grid and values.
    inline CELER_FUNCTION GridIdFinder(SpanConstGrid, SpanConstValue);

    // Find the given grid point
    inline CELER_FUNCTION result_type operator()(argument_type arg) const;

  private:
    SpanConstGrid  grid_;
    SpanConstValue value_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "GridIdFinder.i.hh"
