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
    using SpanConstDbl = Span<double const>;
    using SpanConstFlt = Span<float const>;
    //!@}

  public:
    // Construct with a reference to mutable host data
    XsGridInserter(Values* reals, GridValues* grids);

    // Add a grid of xs-like data
    GridId operator()(UniformGridData const& lower_grid,
                      SpanConstDbl lower_values,
                      UniformGridData const& upper_grid,
                      SpanConstDbl upper_values);
    GridId operator()(UniformGridData const& lower_grid,
                      SpanConstFlt lower_values,
                      UniformGridData const& upper_grid,
                      SpanConstFlt upper_values);

    // Add a grid of uniform log-grid data
    GridId operator()(UniformGridData const& grid, SpanConstDbl values);
    GridId operator()(UniformGridData const& grid, SpanConstFlt values);

  private:
    DedupeCollectionBuilder<real_type> reals_;
    CollectionBuilder<XsGridRecord, MemSpace::host, GridId> grids_;

    template<class T>
    GridId insert(UniformGridData const&,
                  Span<T const>,
                  UniformGridData const&,
                  Span<T const>);
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
