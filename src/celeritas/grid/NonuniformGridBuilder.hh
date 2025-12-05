//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/NonuniformGridBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <utility>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/DedupeCollectionBuilder.hh"
#include "corecel/grid/NonuniformGridData.hh"
#include "corecel/inp/Grid.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct a nonuniform grid.
 *
 * This uses a deduplicating inserter for real values to improve caching.
 */
class NonuniformGridBuilder
{
  public:
    //!@{
    //! \name Type aliases
    using Grid = NonuniformGridRecord;
    using Values = Collection<real_type, Ownership::value, MemSpace::host>;
    //!@}

  public:
    // Construct with a reference to mutable host data
    explicit NonuniformGridBuilder(Values* reals);

    // Add an imported physics vector as a grid
    Grid operator()(inp::Grid const&);

  private:
    Values* values_;
    DedupeCollectionBuilder<real_type> reals_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
