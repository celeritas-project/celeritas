//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/TwodGridBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/DedupeCollectionBuilder.hh"
#include "corecel/grid/TwodGridData.hh"
#include "celeritas/inp/Grid.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct a nonuniform 2D grid.
 *
 * This uses a deduplicating inserter for real values to improve caching.
 */
class TwodGridBuilder
{
  public:
    //!@{
    //! \name Type aliases
    using Values = Collection<real_type, Ownership::value, MemSpace::host>;
    using TwodGrid = TwodGridData;
    //!@}

  public:
    // Construct with pointers to data that will be modified
    explicit TwodGridBuilder(Values* reals);

    // Add a grid from an imported physics vector
    TwodGrid operator()(inp::TwodGrid const&);

  private:
    DedupeCollectionBuilder<real_type> reals_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
