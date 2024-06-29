//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/GenericGridInserter.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/Span.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/DedupeCollectionBuilder.hh"
#include "celeritas/io/ImportPhysicsVector.hh"

#include "GenericGridData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Stores generic, linearly interpolated grids data in a single scalar
 * collection.
 *
 * Only stores the grid and values in a scalar collection, returing the
 * grid itself.
 */
class GenericGridSingleInserter
{
  public:
    //!@{
    //! \name Type aliases
    using RealCollection
        = Collection<real_type, Ownership::value, MemSpace::host>;
    using SpanConstFlt = Span<float const>;
    using SpanConstDbl = Span<double const>;
    //!}

  public:
    //! Construct with collection to be populated
    explicit GenericGridSingleInserter(RealCollection* reals);

    //! Add a grid of generic data with linear interpolation
    GenericGridData operator()(SpanConstFlt grid, SpanConstFlt values);

    //! Add a grid of generic data with linear interpolation
    GenericGridData operator()(SpanConstDbl grid, SpanConstDbl values);

    //! Add an imported physics vector as a generic grid
    GenericGridData operator()(ImportPhysicsVector const& vec);

    //! Add an empty grid (no data present)
    GenericGridData operator()();

  private:
    DedupeCollectionBuilder<real_type> reals_;

    //! Add a grid with appropriate type conversion
    template<class T>
    GenericGridData insert_impl(Span<T const> grid, Span<T const> values);
};

//---------------------------------------------------------------------------//
/*!
 * Stores generic, linearly interpolated grids in a collection.
 *
 * Stores both the grid as scalars in a collection, as well as the grid
 * in a collection indexed by the template parameter.
 */
template<class Index>
class GenericGridInserter
{
  public:
    //!@{
    //! \name Type aliases
    using RealCollection
        = Collection<real_type, Ownership::value, MemSpace::host>;
    using GridCollection
        = Collection<GenericGridData, Ownership::value, MemSpace::host, Index>;
    using SpanConstFlt = Span<float const>;
    using SpanConstDbl = Span<double const>;
    using GridId = Index;
    //!@}

  public:
    //! Construct with collections to be populated
    explicit GenericGridInserter(RealCollection* reals, GridCollection* grids);

    //! Add a grid of generic data with linear interpolation
    Index operator()(SpanConstFlt grid, SpanConstFlt values);

    //! Add a grid of generic data with linear interpolation
    Index operator()(SpanConstDbl grid, SpanConstDbl values);

    //! Add an imported physics vector as a generic grid
    Index operator()(ImportPhysicsVector const& vec);

    //! Add an empty grid (no data present)
    Index operator()();

  private:
    GenericGridSingleInserter single_inserter_;
    CollectionBuilder<GenericGridData, MemSpace::host, Index> grids_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with collections to be populated.
 */
template<class Index>
GenericGridInserter<Index>::GenericGridInserter(RealCollection* reals,
                                                GridCollection* grids)
    : single_inserter_(reals), grids_(grids)
{
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid of generic data with linear interpolation.
 */
template<class Index>
Index GenericGridInserter<Index>::operator()(SpanConstFlt grid,
                                             SpanConstFlt values)
{
    return grids_.push_back(single_inserter_(grid, values));
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid of generic data with linear interpolation.
 */
template<class Index>
Index GenericGridInserter<Index>::operator()(SpanConstDbl grid,
                                             SpanConstDbl values)
{
    return grids_.push_back(single_inserter_(grid, values));
}

//---------------------------------------------------------------------------//
/*!
 * Add an imported physics vector as a generic grid.
 */
template<class Index>
Index GenericGridInserter<Index>::operator()(ImportPhysicsVector const& vec)
{
    return grids_.push_back(single_inserter_(vec));
}

//---------------------------------------------------------------------------//
/*!
 * Add an empty grid (no data present).
 */
template<class Index>
Index GenericGridInserter<Index>::operator()()
{
    return grids_.push_back(single_inserter_());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
