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
 * Class for storing offset data for RaggedRightIndexer
 */
template<size_type N>
class RaggedRightIndexerData
{
  public:
    //!@{
    //! \name Type aliases
    using Sizes = Array<size_type, N>;
    using Offsets = Array<size_type, N + 1>;
    //!@}

  public:
    //! Construct with the an array denoting the size of each dimension.
    explicit CELER_FORCEINLINE_FUNCTION RaggedRightIndexerData(Sizes sizes)
    {
        CELER_EXPECT(sizes.size() > 0);
        offsets_[0] = 0;

        for (auto i : range(N))
        {
            CELER_EXPECT(sizes[i] > 0);
            offsets_[i + 1] = sizes[i] + offsets_[i];
        }
    }

    //! Default for unassigned/lazy construction
    RaggedRightIndexerData() = default;

    //! Access offsets
    CELER_FORCEINLINE_FUNCTION Offsets const& offsets() const
    {
        return offsets_;
    }

  private:
    Offsets offsets_;
};

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
template<size_type N>
class RaggedRightIndexer
{
  public:
    //!@{
    //! \name Type aliases
    using Coords = Array<size_type, 2>;
    //!@}

  public:
    // Construct from RaggedRightIndexerData
    explicit inline CELER_FUNCTION
    RaggedRightIndexer(RaggedRightIndexerData<N> const& rrd);

    //// METHODS ////

    // Convert ragged indices to a flattened index
    inline CELER_FUNCTION size_type index(Coords coords) const;

    // Convert a flattened index into ragged indices
    inline CELER_FUNCTION Coords coords(size_type index) const;

  private:
    //// DATA ////

    RaggedRightIndexerData<N> const& rrd_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct fom RaggedRightIndexerData
 */
template<size_type N>
CELER_FUNCTION
RaggedRightIndexer<N>::RaggedRightIndexer(RaggedRightIndexerData<N> const& rrd)
    : rrd_(rrd)
{
}

//---------------------------------------------------------------------------//
/*!
 * Convert ragged indices to a flattened index.
 */
template<size_type N>
CELER_FUNCTION size_type RaggedRightIndexer<N>::index(Coords ri) const
{
    auto const& offsets = rrd_.offsets();
    CELER_EXPECT(ri[0] < N);
    CELER_EXPECT(ri[1] < offsets[ri[0] + 1] - offsets[ri[0]]);

    return offsets[ri[0]] + ri[1];
}

//---------------------------------------------------------------------------//
/*!
 * Convert a flattened index into ragged indices.
 */
template<size_type N>
CELER_FUNCTION typename RaggedRightIndexer<N>::Coords
RaggedRightIndexer<N>::coords(size_type index) const
{
    auto const& offsets = rrd_.offsets();
    CELER_EXPECT(index < offsets.back());
    CELER_EXPECT(index >= 0);

    size_type i = 0;
    while (index >= offsets[i + 1])
    {
        ++i;
    }

    return Coords{i, index - offsets[i]};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
