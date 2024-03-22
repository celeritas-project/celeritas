//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/GenericGridInserter.cc
//---------------------------------------------------------------------------//
#include "GenericGridInserter.hh"

#include "celeritas/io/ImportPhysicsVector.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a reference to mutable host data.
 */
GenericGridInserter::GenericGridInserter(RealCollection* real_data,
                                         GenericGridCollection* grid)
    : grid_builder_(real_data), grids_(grid)
{
    CELER_EXPECT(real_data && grid);
}

//---------------------------------------------------------------------------//
/*!
 * Add an imported physics vector as a generic grid to the collection.
 * Returns the id of the inserted grid, or an empty id if the vector is empty.
 */
auto GenericGridInserter::operator()(ImportedPhysicsVector const& vec) -> GenericIndex
{
    if (vec.x.empty())
        return GenericIndex{};

    return grids_.push_back(grid_builder_(vec));
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid of generic data with linear interpolation to the collection.
 */
auto GenericGridInserter::operator()(SpanConstFlt grid, SpanConstFlt values) -> GenericIndex
{
    if (grid.empty())
        return GenericIndex{};

    return grids_.push_back(grid_builder_(grid, values));
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid of generic data with linear interpolation to the collection.
 */
auto GenericGridInserter::operator()(SpanConstDbl grid, SpanConstDbl values) -> GenericIndex
{
    if (grid.empty())
        return GenericIndex{};

    return grids_.push_back(grid_builder_(grid, values));
}

//---------------------------------------------------------------------------//
/*!
 * Add an empty grid. Useful for when there's no imported grid present for
 * a given material.
 */
auto GenericGridInserter::operator()(void) -> GenericIndex
{
    return grids_.push_back({});
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
