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
    using BC = SplineDerivCalculator::BoundaryCondition;
    using Grid = NonuniformGridRecord;
    using SpanConstFlt = Span<float const>;
    using SpanConstDbl = Span<double const>;
    //!@}

  public:
    // Construct with pointers to data that will be modified
    explicit NonuniformGridBuilder(Items<real_type>* reals);

    // Construct with pointers to data and spline boundary conditions
    NonuniformGridBuilder(Items<real_type>* reals, BC bc);

    // Add a grid of generic data with linear interpolation
    Grid operator()(SpanConstFlt grid, SpanConstFlt values);

    // Add a grid of generic data with linear interpolation
    Grid operator()(SpanConstDbl grid, SpanConstDbl values);

    // Add a grid from an imported physics vector
    Grid operator()(ImportPhysicsVector const&);

  private:
    Items<real_type> const& values_;
    DedupeCollectionBuilder<real_type> reals_;
    BC bc_;

    // Insert with floating point conversion if needed
    template<class T>
    Grid insert_impl(Span<T const> grid, Span<T const> values);
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
