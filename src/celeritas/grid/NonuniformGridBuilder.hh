//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/NonuniformGridBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/DedupeCollectionBuilder.hh"
#include "corecel/grid/NonuniformGridData.hh"
#include "celeritas/inp/Physics.hh"

namespace celeritas
{
struct ImportPhysicsVector;
//---------------------------------------------------------------------------//
/*!
 * Construct a generic grid.
 *
 * This uses a deduplicating inserter for real values to improve cacheing.
 */
class NonuniformGridBuilder
{
  public:
    //!@{
    //! \name Type aliases
    template<class T>
    using Items = Collection<T, Ownership::value, MemSpace::host>;
    using Grid = NonuniformGridRecord;
    using SpanConstFlt = Span<float const>;
    using SpanConstDbl = Span<double const>;
    //!@}

  public:
    // Construct with pointers to data that will be modified
    explicit NonuniformGridBuilder(Items<real_type>* reals);

    // Add a grid of generic data with linear interpolation
    Grid operator()(SpanConstFlt grid, SpanConstFlt values);
    Grid operator()(SpanConstDbl grid, SpanConstDbl values);

    // Add a grid of generic data with interpolation method
    Grid operator()(SpanConstFlt grid, SpanConstFlt values, inp::Interpolation);
    Grid operator()(SpanConstDbl grid, SpanConstDbl values, inp::Interpolation);

    // Add a grid from an imported physics vector with linear interpolation
    Grid operator()(ImportPhysicsVector const&);

    // Add a grid from an imported physics vector with interpolation method
    Grid operator()(ImportPhysicsVector const&, inp::Interpolation);

  private:
    Items<real_type> const& values_;
    DedupeCollectionBuilder<real_type> reals_;

    // Insert with floating point conversion if needed
    template<class T>
    Grid
    insert_impl(Span<T const> grid, Span<T const> values, inp::Interpolation);
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
