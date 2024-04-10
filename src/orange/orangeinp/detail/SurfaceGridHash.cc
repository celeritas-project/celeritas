//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/SurfaceGridHash.cc
//---------------------------------------------------------------------------//
#include "SurfaceGridHash.hh"

#include <cmath>
#include <iomanip>

#include "corecel/io/Logger.hh"
#include "corecel/io/Repr.hh"
#include "corecel/math/HashUtils.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with grid width.
 *
 * To calculate the grid bin in a way to minimize spatial collisions, we add an
 * offset so that values near zero map to the same bin.
 */
SurfaceGridHash::SurfaceGridHash(real_type grid_scale, real_type tol)
    : eps_{tol}, grid_offset_{grid_scale * 0.5}, inv_grid_width_{1 / grid_scale}
{
    CELER_EXPECT(eps_ > 0);
    CELER_EXPECT(grid_offset_ < 1 / inv_grid_width_);
    CELER_EXPECT(1 / inv_grid_width_ > 2 * eps_);

    CELER_LOG(debug) << "Grid offset: " << grid_offset_
                     << ", inverse width: " << inv_grid_width_;
}

//---------------------------------------------------------------------------//
/*!
 * Update the capacity based on an expected number of surfaces.
 */
void SurfaceGridHash::reserve(size_type cap)
{
    surfaces_.reserve(cap);
}

//---------------------------------------------------------------------------//
/*!
 * Insert a new surface and hash point.
 */
auto SurfaceGridHash::insert(SurfaceType type,
                             real_type hash_point,
                             LocalSurfaceId lsid) -> Insertion
{
    CELER_EXPECT(lsid);

    // Insert the actual surface
    Insertion result;
    auto const orig_bin = this->calc_bin(type, hash_point);
    result.first = surfaces_.insert({orig_bin, lsid});

    // Bump the hash point to the left
    if (auto bin = this->calc_bin(type, hash_point - eps_); bin != orig_bin)
    {
        // Left point is in a different bin
        result.second = surfaces_.insert({bin, lsid});
    }
    else if (auto bin = this->calc_bin(type, hash_point + eps_);
             bin != orig_bin)
    {
        // Right point is in a different bin
        result.second = surfaces_.insert({bin, lsid});
    }
    else
    {
        // Left and right are both in the same bin as the actual point
        result.second = surfaces_.end();
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Erase an added surface.
 *
 * The intended use case is after adding a surface (which returns one or two
 * iterators) the surface can be deleted if it's an exact duplicate.
 */
void SurfaceGridHash::erase(iterator it)
{
    CELER_EXPECT(it != surfaces_.end());
    surfaces_.erase(it);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the hash of a new data point.
 *
 * This combines a hashed value of the unique grid bin with the surfaces by
 * \em replacing the lowest bits with the surface type representation. <em>This
 * is necessary to ensure the same "value" has surfaces that have all the same
 * type.</em>
 */
auto SurfaceGridHash::calc_bin(SurfaceType type, real_type hash_point) const
    -> key_type
{
    auto grid_bin = std::floor((hash_point + grid_offset_) * inv_grid_width_);
    auto hash = hash_combine(grid_bin);
    static_assert(std::is_same_v<decltype(hash), size_type>);

    // Clear the lowest 5 bits
    static_assert(static_cast<size_type>(SurfaceType::size_) < 32);
    hash &= (~static_cast<size_type>(0x50 - 1));
    hash |= static_cast<size_type>(type);

    CELER_LOG(debug) << "Hashed " << to_cstring(type) << "@"
                     << repr(hash_point) << " (bin " << grid_bin << ") -> "
                     << std::hex << hash;

    return hash;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
