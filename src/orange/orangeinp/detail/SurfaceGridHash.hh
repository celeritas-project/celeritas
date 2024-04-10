//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/SurfaceGridHash.hh
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>
#include <cstdlib>
#include <unordered_map>
#include <utility>

#include "orange/OrangeTypes.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Store a hash of "similar" surfaces for faster lookups.
 *
 * This stores local surface indices on an infinite grid.
 * - Nearby surfaces should always have nearby "hash points", within some
 *   comparison tolerance
 * - The comparison tolerance must be less than the grid width, probably \em
 *   much less
 * - Different surfaces can have an identical hash point but have
 *   different surface types
 * - A range of surfaces from the \c find method will always match the given
 *   surface type.
 */
class SurfaceGridHash
{
  public:
    //!@{
    //! \name Type aliases
    using size_type = std::size_t;
    using key_type = size_type;
    using mapped_type = LocalSurfaceId;
    using MultimapSurfaces = std::unordered_multimap<key_type, mapped_type>;
    using iterator = MultimapSurfaces::iterator;
    using const_iterator = MultimapSurfaces::const_iterator;
    using Insertion = std::pair<iterator, iterator>;
    using ConstRange = std::pair<const_iterator, const_iterator>;
    //!@}

  public:
    // Construct with maximum tolerance and characteristic scale of grid
    SurfaceGridHash(real_type grid_scale, real_type tol);

    // Update the capacity based on an expected number of surfaces
    void reserve(size_type);

    // Insert a new surface and hash point
    Insertion insert(SurfaceType type, real_type hash_point, LocalSurfaceId);

    // Erase an added surface
    void erase(iterator);

    //! Get the begin iterator for comparing with insertion
    const_iterator begin() const { return surfaces_.begin(); }

    //! Get the end iterator for comparing with insertion
    const_iterator end() const { return surfaces_.end(); }

    // Find all local surfaces that have the same bin (hash + type)
    inline ConstRange equal_range(const_iterator iter) const;

  private:
    real_type eps_;
    real_type grid_offset_;
    real_type inv_grid_width_;
    MultimapSurfaces surfaces_;

    // Calculate the bin of a new data point
    key_type calc_bin(SurfaceType type, real_type hash_point) const;
};

//---------------------------------------------------------------------------//
/*!
 * Find all local surfaces that have the same bin (hash + type).
 */
auto SurfaceGridHash::equal_range(const_iterator iter) const -> ConstRange
{
    if (iter == this->end())
    {
        return {iter, iter};
    }

    // TODO: we can't search based on the iterator's bucket... this is going to
    // lose some efficiency by re-hashing the key and doing another search.
    return surfaces_.equal_range(iter->first);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
