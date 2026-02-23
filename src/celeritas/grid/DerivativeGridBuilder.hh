//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/DerivativeGridBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/Collection.hh"
#include "corecel/data/DedupeCollectionBuilder.hh"
#include "corecel/grid/NonuniformGridData.hh"
#include "corecel/inp/Grid.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct a derivative grid from the given grid record.
 *
 * Since grids are piecewise function definitions, the derivative might not be
 * well defined in a region around a grid-point. An epsilon-neighborhood is
 * constructed around each grid-point where the derivative may be interpolated
 * between the two derivative grid-points. Outside of this neighborhood the
 * derivative is well defined since the interpolation function is smooth.
 *
 * A deduplicating inserter for real values is used to improve caching.
 *
 * \todo Currently only \c NonuniformGridRecord is supported since it is
 * necessary for calculating group velocity from refractive index.
 */
class DerivativeGridBuilder
{
  public:
    //!@{
    //! \name Type aliases
    using Grid = NonuniformGridRecord;
    using Values = Collection<real_type, Ownership::value, MemSpace::host>;
    //!@}

  public:
    // Construct with a reference to mutable host data
    explicit DerivativeGridBuilder(Values* reals, real_type epsilon);

    // Construct the derivative grid of an imported grid
    Grid operator()(inp::Grid const&);

  private:
    real_type epsilon_;
    Values* values_;
    DedupeCollectionBuilder<real_type> reals_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
