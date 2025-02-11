//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/ValueGridInserter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <utility>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/grid/UniformGridData.hh"
#include "celeritas/Types.hh"

#include "XsGridData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Manage data and help construction of physics value grids.
 *
 * Currently this only constructs a single value grid datatype, the
 * XsGridData, but with this framework (virtual \c
 * ValueGridXsBuilder::build method taking an instance of this class) it can be
 * extended to build additional grid types as well.
 *
 * \code
    ValueGridInserter insert(&data.host.values, &data.host.grids);
    insert(uniform_grid, values);
    store.push_back(host_ptrs);
    store.copy_to_device();
   \endcode
 */
class ValueGridInserter
{
  public:
    //!@{
    //! \name Type aliases
    using Values = Collection<real_type, Ownership::value, MemSpace::host>;
    using GridValues = Collection<XsGridData, Ownership::value, MemSpace::host>;
    using SpanConstDbl = Span<double const>;
    using SpanConstFlt = Span<float const>;
    using XsIndex = ItemId<XsGridData>;
    //!@}

  public:
    // Construct with a reference to mutable host data
    ValueGridInserter(Values* reals, GridValues* grids);

    // Add a grid of xs-like data
    XsIndex operator()(UniformGridData const& log_grid,
                       size_type prime_index,
                       SpanConstDbl values);
    XsIndex operator()(UniformGridData const& log_grid,
                       size_type prime_index,
                       SpanConstFlt values);

    // Add a grid of uniform log-grid data
    XsIndex operator()(UniformGridData const& log_grid, SpanConstDbl values);
    XsIndex operator()(UniformGridData const& log_grid, SpanConstFlt values);

  private:
    CollectionBuilder<real_type, MemSpace::host, ItemId<real_type>> values_;
    CollectionBuilder<XsGridData, MemSpace::host, ItemId<XsGridData>> xs_grids_;

    template<class T>
    XsIndex insert_xs(UniformGridData const&, size_type, Span<T const>);
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
