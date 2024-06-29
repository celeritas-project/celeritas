//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/GenericGridBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/cont/Span.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Intermediate storage for generic grids and their associated values.
 *
 * Matches the behavior of ValueGridBuilder, with the intention of
 * constructing builders independently of allocating and populating the
 * host collection where the data will reside during processing.
 */
class GenericGridBuilder
{
  public:
    //!@{
    //! \name Type aliases
    using SpanConstDbl = Span<double const>;
    //!@}

  public:
    //! Construct the builder from imported Geant grids
    static std::unique_ptr<GenericGridBuilder>
    from_geant(SpanConstDbl grid, SpanConstDbl values);

    //! Construct the builder directly from grids
    GenericGridBuilder(SpanConstDbl grid, SpanConstDbl values);

    //! Build the grid in the given store
    template<class Inserter>
    inline typename Inserter::GridId build(Inserter insert) const;

  private:
    SpanConstDbl grid_, values_;

    //!@{
    //! Prevent copy/move expect by daughters that know what they're doing
    GenericGridBuilder(GenericGridBuilder const&) = default;
    GenericGridBuilder& operator=(GenericGridBuilder const&) = default;
    GenericGridBuilder(GenericGridBuilder&&) = default;
    GenericGridBuilder& operator=(GenericGridBuilder&&) = default;
    //!@}
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Build the grid in the given store.
 *
 * This is primarily intended to be used with GenericGridInserters that are
 * templated against a specific opaque ID.
 */
template<class Inserter>
auto GenericGridBuilder::build(Inserter insert) const ->
    typename Inserter::GridId
{
    return insert(grid_, values_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
