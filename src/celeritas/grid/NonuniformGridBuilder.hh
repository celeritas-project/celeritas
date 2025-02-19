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
#include "corecel/grid/SplineDerivCalculator.hh"

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
    using BC = SplineDerivCalculator::BoundaryCondition;
    using SpanConstFlt = Span<float const>;
    using SpanConstDbl = Span<double const>;
    //!@}

  public:
    // Construct with pointers to data that will be modified
    explicit NonuniformGridBuilder(Items<real_type>* reals);

    // Add a grid of generic data with linear interpolation
    Grid operator()(SpanConstFlt grid, SpanConstFlt values);
    Grid operator()(SpanConstDbl grid, SpanConstDbl values);

    // Add a grid of generic data with spline interpolation
    Grid operator()(SpanConstFlt grid, SpanConstFlt values, BC bc);
    Grid operator()(SpanConstDbl grid, SpanConstDbl values, BC bc);

    // Add a grid from an imported physics vector with linear interpolation
    Grid operator()(ImportPhysicsVector const&);

    // Add a grid from an imported physics vector with spline interpolation
    Grid operator()(ImportPhysicsVector const&, BC bc);

  private:
    Items<real_type> const& values_;
    DedupeCollectionBuilder<real_type> reals_;

    // Insert with floating point conversion if needed
    template<class T>
    Grid insert_impl(Span<T const> grid, Span<T const> values, BC bc);
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
