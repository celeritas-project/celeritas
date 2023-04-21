//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/RaggedRightIndexer.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Range.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Class for indexing into flattened, ragged-right, 2D data.
 *
 * For example, consider three arrays of different sizes:
 *  A = [a1, a2]
 *  B = [b1, b2, b3]
 *  C = [c1]
 *
 *  Flattening them into a single array gives
 *
 *  flattened = [a1, a2, b1, b2, b3, c1]
 *
 *  Within this array, element b3 has a "flattened" index of 4 and "ragged
 * indices" of [1, 2]
 */
template<class T, size_type N>
class RaggedRightIndexer
{
  public:
    //!@{
    //! \name Type aliases
    using index_type = T;
    using Sizes = Array<index_type, N>;
    using RaggedIndices = Array<index_type, 2>;
    //!@}

  public:
    // Construct with the an array denoting the size of each dimension
    explicit inline CELER_FUNCTION RaggedRightIndexer(Sizes const& sizes);

    //// METHODS ////

    //! Convert ragged indices to a flattened index
    inline CELER_FUNCTION index_type
    flattened_index(RaggedIndices ragged_indices) const;

    //! Convert a flattened index into ragged indices
    inline CELER_FUNCTION RaggedIndices
    ragged_indices(index_type flattened_index) const;

  private:
    //// DATA ////

    Array<index_type, N + 1> offsets_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from array denoting the size of each dimension.
 */
template<class T, size_type N>
CELER_FUNCTION RaggedRightIndexer<T, N>::RaggedRightIndexer(Sizes const& sizes)
{
    CELER_EXPECT(sizes.size() > 0);

    offsets_[0] = 0;

    for (auto i : range(N))
    {
        CELER_EXPECT(sizes[i] > 0);
        offsets_[i + 1] = sizes[i] + offsets_[i];
    }
}

//---------------------------------------------------------------------------//
/*!
 * Convert ragged indices to a flattened index.
 */
template<class T, size_type N>
CELER_FUNCTION typename RaggedRightIndexer<T, N>::index_type
RaggedRightIndexer<T, N>::flattened_index(RaggedIndices ri) const
{
    CELER_EXPECT(ri[0] < N);
    CELER_EXPECT(ri[1] < offsets_[ri[0] + 1] - offsets_[ri[0]]);

    return offsets_[ri[0]] + ri[1];
}

//---------------------------------------------------------------------------//
/*!
 * Convert a flattened index into ragged indices.
 */
template<class T, size_type N>
CELER_FUNCTION typename RaggedRightIndexer<T, N>::RaggedIndices
RaggedRightIndexer<T, N>::ragged_indices(index_type flattened_index) const
{
    CELER_EXPECT(flattened_index < offsets_.back());
    CELER_EXPECT(flattened_index >= 0);

    index_type i = 0;
    while (flattened_index >= offsets_[i + 1])
    {
        ++i;
    }

    return RaggedIndices{i, flattened_index - offsets_[i]};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
