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
 * Class for storing offset data for RaggedRightIndexer.
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
 * Index into flattened, ragged-right, 2D data, from index to coords
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
 *  Within this array, index of 4 (element b3) returns coords [1, 2].
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
    inline CELER_FUNCTION size_type operator()(Coords coords) const;

  private:
    //// DATA ////

    RaggedRightIndexerData<N> const& rrd_;
};

//---------------------------------------------------------------------------//
/*!
 * Index into flattened, ragged-right, 2D data, from coords to index
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
 *  Within this array, coords [1, 2] (element b3) returns index 4.
 */
template<size_type N>
class RaggedRightInverseIndexer
{
  public:
    //!@{
    //! \name Type aliases
    using Coords = Array<size_type, 2>;
    //!@}

  public:
    // Construct from RaggedRightIndexerData
    explicit inline CELER_FUNCTION
    RaggedRightInverseIndexer(RaggedRightIndexerData<N> const& rrd);

    //// METHODS ////

    // Convert a flattened index into ragged indices
    inline CELER_FUNCTION Coords operator()(size_type index) const;

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
CELER_FUNCTION size_type RaggedRightIndexer<N>::operator()(Coords coords) const
{
    auto const& offsets = rrd_.offsets();
    CELER_EXPECT(coords[0] < N);
    CELER_EXPECT(coords[1] < offsets[coords[0] + 1] - offsets[coords[0]]);

    return offsets[coords[0]] + coords[1];
}

//---------------------------------------------------------------------------//
/*!
 * Construct fom RaggedRightIndexerData
 */
template<size_type N>
CELER_FUNCTION RaggedRightInverseIndexer<N>::RaggedRightInverseIndexer(
    RaggedRightIndexerData<N> const& rrd)
    : rrd_(rrd)
{
}

//---------------------------------------------------------------------------//
/*!
 * Convert a flattened index into ragged indices.
 */
template<size_type N>
CELER_FUNCTION typename RaggedRightInverseIndexer<N>::Coords
RaggedRightInverseIndexer<N>::operator()(size_type index) const
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
