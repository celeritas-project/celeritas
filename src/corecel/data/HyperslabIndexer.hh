//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/HyperslabIndexer.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Class for indexing into flattened N-dimensional data.
 *
 * Indexing is in standard C iteration order, such that final dimension
 * "changes fastest". For example, when indexing into a 3D grid (N=3) with
 * (i=0, j=0, k=1) the resulting index will be 1.
 */
template<class T, size_type N>
class HyperslabIndexer
{
  public:
    //!@{
    //! \name Type aliases
    using index_type = T;
    using Coords = Array<size_type, N>;
    //!@}

  public:
    // Construct with an array denoting the size of each dimension
    explicit inline CELER_FUNCTION HyperslabIndexer(Coords const& dims);

    //// METHODS ////

    //! Convert N-dimensional coordinates to an index
    inline CELER_FUNCTION index_type index(Coords const& coords) const;

    //! Convert an index to N-dimensional coordinates
    inline CELER_FUNCTION Coords coords(index_type index) const;

  private:
    //// DATA ////

    Coords dims_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from array denoting the sizes of each dimension.
 */
template<class T, size_type N>
CELER_FUNCTION HyperslabIndexer<T, N>::HyperslabIndexer(Coords const& dims)
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
template<class T, size_type N>
CELER_FUNCTION typename HyperslabIndexer<T, N>::index_type
HyperslabIndexer<T, N>::index(Coords const& coords) const
{
    index_type index = 0;
    index_type offset = 1;
    for (int i = dims_.size() - 1; i >= 0; i--)
    {
        index += coords[i] * offset;
        offset *= dims_[i];
    }

    return index;
}

//---------------------------------------------------------------------------//
/*!
 * Convert an index into N-dimensional coordinates.
 */
template<class T, size_type N>
CELER_FUNCTION typename HyperslabIndexer<T, N>::Coords
HyperslabIndexer<T, N>::coords(index_type index) const
{
    Coords coords;

    for (int i = dims_.size() - 1; i >= 0; i--)
    {
        coords[i] = index % dims_[i];
        index = (index - coords[i]) / dims_[i];
    }

    return coords;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
