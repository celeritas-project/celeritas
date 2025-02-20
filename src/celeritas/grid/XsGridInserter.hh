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
#include "celeritas/inp/Physics.hh"

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
    GridId operator()(UniformGridData const&,
                      SpanConstDbl,
                      inp::Interpolation,
                      UniformGridData const&,
                      SpanConstDbl,
                      inp::Interpolation);
    GridId operator()(UniformGridData const&,
                      SpanConstFlt,
                      inp::Interpolation,
                      UniformGridData const&,
                      SpanConstFlt,
                      inp::Interpolation);

    // Add a grid of uniform log-grid data
    GridId operator()(UniformGridData const&, SpanConstDbl, inp::Interpolation);
    GridId operator()(UniformGridData const&, SpanConstFlt, inp::Interpolation);

  private:
    using ValuesRef
        = Collection<real_type, Ownership::const_reference, MemSpace::host>;

    Values const& values_;
    DedupeCollectionBuilder<real_type> reals_;
    CollectionBuilder<XsGridRecord, MemSpace::host, GridId> grids_;

    template<class T>
    GridId insert(UniformGridData const&,
                  Span<T const>,
                  inp::Interpolation,
                  UniformGridData const&,
                  Span<T const>,
                  inp::Interpolation);
    void set_spline(UniformGridData const&,
                    inp::Interpolation const&,
                    UniformGridRecord&);
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
