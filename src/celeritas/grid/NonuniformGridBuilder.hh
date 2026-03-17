//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/NonuniformGridBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/DedupeCollectionBuilder.hh"
#include "corecel/grid/NonuniformGridData.hh"
#include "corecel/inp/Grid.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct a nonuniform grid.
 *
 * This uses a deduplicating inserter for real values to improve caching.
 *
 * \todo Move to corecel/grid
 * \todo Take grid by capturing and eliminate duplicate points.
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
