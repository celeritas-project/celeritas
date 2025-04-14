//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/XsGridInserter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <utility>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/DedupeCollectionBuilder.hh"
#include "corecel/grid/UniformGridData.hh"
#include "celeritas/Types.hh"
#include "celeritas/inp/Grid.hh"

#include "XsGridData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Manage data and help construction of physics cross section grids.
 */
class XsGridInserter
{
  public:
    //!@{
    //! \name Type aliases
    using GridId = ItemId<XsGridRecord>;
    using GridValues
        = Collection<XsGridRecord, Ownership::value, MemSpace::host>;
    using Values = Collection<real_type, Ownership::value, MemSpace::host>;
    //!@}

  public:
    // Construct with a reference to mutable host data
    XsGridInserter(Values* reals, GridValues* grids);

    // Add a grid of xs-like data
    GridId
    operator()(inp::UniformGrid const& lower, inp::UniformGrid const& upper);

    // Add a grid of uniform log-grid data
    GridId operator()(inp::UniformGrid const& grid);

  private:
    DedupeCollectionBuilder<real_type> reals_;
    CollectionBuilder<XsGridRecord, MemSpace::host, GridId> grids_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
