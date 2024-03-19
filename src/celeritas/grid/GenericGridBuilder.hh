//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/GenericGridBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/DedupeCollectionBuilder.hh"

#include "GenericGridData.hh"

namespace celeritas
{
struct ImportPhysicsVector;
//---------------------------------------------------------------------------//
/*!
 * Construct a generic grid.
 *
 * This uses a deduplicating inserter for real values to improve cacheing.
 */
class GenericGridBuilder
{
  public:
    //!@{
    //! \name Type aliases
    template<class T>
    using Items = Collection<T, Ownership::value, MemSpace::host>;
    using Grid = GenericGridData;
    using SpanConstReal = Span<real_type const>;
    //!@}

  public:
    // Construct with pointers to data that will be modified
    explicit GenericGridBuilder(Items<real_type>* reals);

    // Add a grid of generic data with linear interpolation
    Grid operator()(SpanConstReal grid, SpanConstReal values);

    // Add a grid from an imported physics vector
    Grid operator()(ImportPhysicsVector const&);

  private:
    DedupeCollectionBuilder<real_type> reals_;

    // Insert via containers for floating point conversions
    template<class Cont>
    Grid insert_impl(Cont const& grid, Cont const& values);
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
