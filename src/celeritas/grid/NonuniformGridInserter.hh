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
#include "corecel/cont/Span.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/Types.hh"
#include "celeritas/io/ImportPhysicsVector.hh"

#include "NonuniformGridBuilder.hh"
#include "NonuniformGridData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct a generic grid using mutable host data and add it to
 * the specified grid collection.
 *
 * \code
    NonuniformGridInserter insert(&data->reals, &data->generic_grids);
    std::vector<NonuniformGridIndex> grid_ids;
    for (material : range(MaterialId{mats->size()}))
        grid_ids.push_back(insert(physics_vector[material.get()]));
   \endcode
 */
template<class Index>
class NonuniformGridInserter
{
  public:
    //!@{
    //! \name Type aliases
    using SpanConstFlt = Span<float const>;
    using SpanConstDbl = Span<double const>;
    using Values = Collection<real_type, Ownership::value, MemSpace::host>;
    using GridValues
        = Collection<NonuniformGridRecord, Ownership::value, MemSpace::host, Index>;
    //!@}

  public:
    // Construct with a reference to mutable host data
    NonuniformGridInserter(Values* reals, GridValues* grids);

    // Add a grid of generic data with linear interpolation
    Index operator()(SpanConstFlt grid, SpanConstFlt values);

    // Add a grid of generic data with linear interpolation
    Index operator()(SpanConstDbl grid, SpanConstDbl values);

    // Add an imported physics vector as a grid
    Index operator()(ImportPhysicsVector const& vec);

    // Add an empty grid (no data present)
    Index operator()();

  private:
    NonuniformGridBuilder grid_builder_;
    CollectionBuilder<NonuniformGridRecord, MemSpace::host, Index> grids_;
};

//---------------------------------------------------------------------------//
/*!
 * Construct with a reference to mutable host data.
 */
template<class Index>
NonuniformGridInserter<Index>::NonuniformGridInserter(Values* reals,
                                                      GridValues* grids)
    : grid_builder_(reals), grids_(grids)
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
auto NonuniformGridInserter<Index>::operator()(ImportPhysicsVector const& vec)
    -> Index
{
    CELER_EXPECT(!vec.x.empty());
    return grids_.push_back(grid_builder_(vec));
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid of generic data with linear interpolation to the collection.
 */
template<class Index>
auto NonuniformGridInserter<Index>::operator()(SpanConstFlt grid,
                                               SpanConstFlt values) -> Index
{
    CELER_EXPECT(!grid.empty());
    return grids_.push_back(grid_builder_(grid, values));
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid of generic data with linear interpolation to the collection.
 */
template<class Index>
auto NonuniformGridInserter<Index>::operator()(SpanConstDbl grid,
                                               SpanConstDbl values) -> Index
{
    CELER_EXPECT(!grid.empty());
    return grids_.push_back(grid_builder_(grid, values));
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
