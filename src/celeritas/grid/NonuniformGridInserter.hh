//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/NonuniformGridInserter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <utility>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/grid/NonuniformGridData.hh"
#include "celeritas/Types.hh"
#include "celeritas/inp/Grid.hh"

#include "NonuniformGridBuilder.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct a nonuniform grid and add it to the specified grid collection.
 */
template<class Index>
class NonuniformGridInserter
{
  public:
    //!@{
    //! \name Type aliases
    using Values = Collection<real_type, Ownership::value, MemSpace::host>;
    using GridValues
        = Collection<NonuniformGridRecord, Ownership::value, MemSpace::host, Index>;
    //!@}

  public:
    // Construct with a reference to mutable host data
    NonuniformGridInserter(Values* reals, GridValues* grids);

    // Add an imported physics vector as a grid
    Index operator()(inp::Grid const&);

    // Add an empty grid (no data present)
    Index operator()();

  private:
    NonuniformGridBuilder build_;
    CollectionBuilder<NonuniformGridRecord, MemSpace::host, Index> grids_;
};

//---------------------------------------------------------------------------//
/*!
 * Construct with a reference to mutable host data.
 */
template<class Index>
NonuniformGridInserter<Index>::NonuniformGridInserter(Values* reals,
                                                      GridValues* grids)
    : build_(reals), grids_(grids)
{
    CELER_EXPECT(reals && grids);
}

//---------------------------------------------------------------------------//
/*!
 * Add an imported physics vector as a generic grid to the collection.
 *
 * Returns the id of the inserted grid, or an empty id if the vector is
 * empty.
 */
template<class Index>
auto NonuniformGridInserter<Index>::operator()(inp::Grid const& grid) -> Index
{
    CELER_EXPECT(!grid.x.empty());
    return grids_.push_back(build_(grid));
}

//---------------------------------------------------------------------------//
/*!
 * Add an empty grid.
 *
 * Useful for when there's no imported grid present for a given material.
 */
template<class Index>
auto NonuniformGridInserter<Index>::operator()() -> Index
{
    return grids_.push_back({});
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
