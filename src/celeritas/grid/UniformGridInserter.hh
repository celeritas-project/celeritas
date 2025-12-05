//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/UniformGridInserter.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/DedupeCollectionBuilder.hh"
#include "corecel/grid/UniformGridData.hh"
#include "corecel/inp/Grid.hh"
#include "celeritas/Types.hh"

#include "XsGridData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Manage data and help construction of physics value grids.
 */
class UniformGridInserter
{
  public:
    //!@{
    //! \name Type aliases
    using GridId = ItemId<UniformGridRecord>;
    using GridValues
        = Collection<UniformGridRecord, Ownership::value, MemSpace::host>;
    using Values = Collection<real_type, Ownership::value, MemSpace::host>;
    //!@}

  public:
    // Construct with a reference to mutable host data
    UniformGridInserter(Values* reals, GridValues* grids);

    // Add a grid of uniform log-grid data
    GridId operator()(inp::UniformGrid const& grid);

  private:
    Values* values_;
    DedupeCollectionBuilder<real_type> reals_;
    CollectionBuilder<UniformGridRecord, MemSpace::host, GridId> grids_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
