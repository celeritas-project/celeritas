//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/GenericGridInserter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <utility>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/Types.hh"

#include "GenericGridBuilder.hh"
#include "GenericGridData.hh"

namespace celeritas
{
struct ImportPhysicsVector;
//---------------------------------------------------------------------------//
/*!
 * Construct a generic grid using mutable host data and add it to
 * the specified grid collection.
 *
 * \code
    GenericGridInserter insert(&data->reals, &data->generic_grids);
    std::vector<GenericGridIndex> grid_ids;
    for (material : range(MaterialId{mats->size()}))
        grid_ids.push_back(insert(physics_vector[material.get()]));
 */
class GenericGridInserter
{
  public:
    //!@{
    //! \name Type aliases
    using RealCollection
        = Collection<real_type, Ownership::value, MemSpace::host>;
    using GenericGridCollection
        = Collection<GenericGridData, Ownership::value, MemSpace::host>;
    using SpanConstFlt = Span<float const>;
    using SpanConstDbl = Span<double const>;
    using GenericIndex = ItemId<GenericGridData>;
    //!@}

  public:
    //! Construct with a reference to mutable host data
    GenericGridInserter(RealCollection* real_data, GenericGridCollection* grid);

    //! Add a grid of generic data with linear interpolation
    GenericIndex operator()(SpanConstFlt grid, SpanConstFlt values);

    //! Add a grid of generic data with linear interpolation
    GenericIndex operator()(SpanConstDbl grid, SpanConstDbl values);

    //! Add an imported physics vector as a grid
    GenericIndex operator()(ImportPhysicsVector const& vec);

    //! Add an empty grid (no data present)
    GenericIndex operator()(void);

  private:
    GenericGridBuilder grid_builder_;
    CollectionBuilder<GenericGridData, MemSpace::host, ItemId<GenericGridData>> grids_;
};
//---------------------------------------------------------------------------//
}  // namespace celeritas
