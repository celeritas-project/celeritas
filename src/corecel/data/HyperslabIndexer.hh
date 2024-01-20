//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/HyperslabIndexer.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"

#include "detail/HyperslabIndexerImpl.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Class for indexing into flattened N-dimensional data (N-D coords to index)
 *
 * Indexing is in standard C iteration order, such that final dimension
 * "changes fastest". For example, when indexing into a 3D grid (N=3) with
 * coords (i=0, j=0, k=1) the resulting index will be 1.
 */
template<size_type N>
class HyperslabIndexer
{
  public:
    //!@{
    //! \name Type aliases
    using Coords = Array<size_type, N>;
    //!@}

  public:
    // Construct with an array denoting the size of each dimension
    explicit inline CELER_FUNCTION
    HyperslabIndexer(Array<size_type, N> const& dims);

    //// METHODS ////

    //! Convert N-dimensional coordinates to an index
    inline CELER_FUNCTION size_type operator()(Coords const& coords) const;

  private:
    //// DATA ////

    Array<size_type, N> const& dims_;
};

//---------------------------------------------------------------------------//
/*!
 * Class for indexing into flattened N-dimensional data (index to N-D coords)
 *
 * Indexing is in standard C iteration order, such that final dimension
 * "changes fastest". For example, when indexing into a 3D grid (N=3), index 1
 * will result in coords (i=0, j=0, k=1).
 */
template<size_type N>
class HyperslabInverseIndexer
{
  public:
    //!@{
    //! \name Type aliases
    using Coords = Array<size_type, N>;
    //!@}

  public:
    // Construct with an array denoting the size of each dimension
    explicit inline CELER_FUNCTION
    HyperslabInverseIndexer(Array<size_type, N> const& dims);

    //// METHODS ////

    //! Convert an index to N-dimensional coordinates
    inline CELER_FUNCTION Coords operator()(size_type index) const;

  private:
    //// DATA ////

    Array<size_type, N> const& dims_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from array denoting the sizes of each dimension.
 */
template<size_type N>
CELER_FUNCTION HyperslabIndexer<N>::HyperslabIndexer(Coords const& dims)
    : dims_(dims)
{
    for (auto dim : dims_)
    {
        CELER_EXPECT(dim > 0);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Convert N-dimensional coordinates to an index.
 */
template<size_type N>
CELER_FUNCTION size_type HyperslabIndexer<N>::operator()(Coords const& coords) const
{
    CELER_EXPECT(coords[0] < dims_[0]);
    size_type result = coords[0];

    for (size_type i = 1; i < N; ++i)
    {
        CELER_EXPECT(coords[i] < dims_[i]);
        result = dims_[i] * result + coords[i];
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct from array denoting the sizes of each dimension.
 */
template<size_type N>
CELER_FUNCTION HyperslabInverseIndexer<N>::HyperslabInverseIndexer(
    Array<size_type, N> const& dims)
    : dims_(dims)
{
    for (auto dim : dims_)
    {
        CELER_EXPECT(dim > 0);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Convert an index into N-dimensional coordinates.
 */
template<size_type N>
CELER_FUNCTION typename HyperslabInverseIndexer<N>::Coords
HyperslabInverseIndexer<N>::operator()(size_type index) const
{
    CELER_EXPECT(index <= detail::hyperslab_size(dims_));
    Coords coords;

    for (size_type i = dims_.size() - 1; i > 0; i--)
    {
        coords[i] = index % dims_[i];
        index = (index - coords[i]) / dims_[i];
    }

    coords[0] = index;

    return coords;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
