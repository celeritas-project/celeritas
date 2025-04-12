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
 * Construct a generic 2D grid.
 *
 * This uses a deduplicating inserter for real values to improve cacheing.
 */
class TwodGridBuilder
{
  public:
    //!@{
    //! \name Type aliases
    template<class T>
    using Items = Collection<T, Ownership::value, MemSpace::host>;
    using TwodGrid = TwodGridData;
    using SpanConstFlt = Span<float const>;
    using SpanConstDbl = Span<double const>;
    //!@}

  public:
    // Construct with pointers to data that will be modified
    explicit TwodGridBuilder(Items<real_type>* reals);

    // Add a 2D grid of generic data with linear interpolation
    TwodGrid
    operator()(SpanConstFlt grid_x, SpanConstFlt grid_y, SpanConstFlt values);

    // Add a 2D grid of generic data with linear interpolation
    TwodGrid
    operator()(SpanConstDbl grid_x, SpanConstDbl grid_y, SpanConstDbl values);

    // Add a grid from an imported physics vector
    TwodGrid operator()(inp::TwodGrid const&);

  private:
    DedupeCollectionBuilder<real_type> reals_;

    // Insert with floating point conversion if needed
    template<class T>
    TwodGrid insert_impl(Span<T const> grid_x,
                         Span<T const> grid_y,
                         Span<T const> values);
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
