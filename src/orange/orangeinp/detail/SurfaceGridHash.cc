//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/SurfaceGridHash.cc
//---------------------------------------------------------------------------//
#include "SurfaceGridHash.hh"

#include <cmath>

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
    : eps_{tol}, grid_offset_{grid_scale / 2}, inv_grid_width_{1 / grid_scale}
{
    CELER_EXPECT(eps_ > 0);
    CELER_EXPECT(grid_offset_ < 1 / inv_grid_width_);
    CELER_EXPECT(1 / inv_grid_width_ > 2 * eps_);
}

//---------------------------------------------------------------------------//
/*!
 * Insert a new surface and hash point.
 */
auto SurfaceGridHash::operator()(SurfaceType type, real_type hash_point) const
    -> result_type
{
    // Insert the actual surface
    result_type result;
    result[0] = this->calc_bin(type, hash_point);

    if (auto second = this->calc_bin(type, hash_point - eps_);
        second != result[0])
    {
        // The hash point bumped to the left is in the same bin
        result[1] = second;
    }
    else if (auto second = this->calc_bin(type, hash_point + eps_);
             second != result[0])
    {
        // Right point is in a different bin
        result[1] = second;
    }
    else
    {
        // Left and right are both in the same bin as the actual point
        result[1] = redundant();
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the hash of a new data point.
 *
 * This combines the grid bin with the surface type by \em replacing the lowest
 * bits with the surface type representation. <em>This guarantees the same
 * "value" has surfaces that have all the same type.</em>
 */
auto SurfaceGridHash::calc_bin(SurfaceType type, real_type hash_point) const
    -> key_type
{
    auto grid_bin = std::floor((hash_point + grid_offset_) * inv_grid_width_);
    auto hash = hash_combine(grid_bin);
    static_assert(std::is_same_v<decltype(hash), key_type>);

    // Clear the lowest 4 bits; make sure there's no valid surface for the
    // "redundant()" value
    static_assert(static_cast<key_type>(SurfaceType::size_) < 0b11111);
    hash &= (~static_cast<key_type>(0b11111));
    hash |= static_cast<key_type>(type);

    return hash;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
