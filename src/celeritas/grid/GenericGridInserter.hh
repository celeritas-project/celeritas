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

#include "corecel/Types.h"
#include "corecel/cont/Span.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/grid/UniformGridData.hh"
#include "celeritas/Types.hh"

#include "GenericGridBuilder.hh"
#include "GenericGridData.hh"

namespace celeritas
{
struct ImportPhysicsVector;
//---------------------------------------------------------------------------//
/*!
 */
class GenericGridInserter
{
  public:
    //!@{
    //! \name Type aliases
    using RealCollection
        = Collection<real_type, Ownership::value, MemSpace::host>;
    using GenericGridCollection
        = Collection<GenericGridData, Ownership::value, Memspace::host>;
    using SpanConstDbl = Span<double const>;
    using GenericIndex = ItemId<GenericGridData>;
    //!@}

  public:
    //! Construct with a reference to mutable host data
    GenericGridInserter(RealCollection* real_data, GenericGridCollection* grid);

    //! Add an imported physics vector as a grid
    GenericIndex operator()(ImportedPhysicsVector const& vec);

  private:
    GenericGridBuilder grid_builder_;
    CollectionBuilder<GenericGridData, MemSpace::host, ItemId<GenericGridData>> grids_;
};
//---------------------------------------------------------------------------//
}  // namespace celeritas
