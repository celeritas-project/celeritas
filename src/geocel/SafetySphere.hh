//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/SafetySphere.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/math/Algorithms.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sphere that is known to contain no geometry boundaries.
 *
 * The sphere is centered on a point where the safety distance was calculated,
 * and its radius is that safety distance. Because a sphere is convex, a line
 * segment whose two end points are inside the sphere is entirely inside it and
 * therefore crosses no boundary.
 *
 * A default-constructed sphere has a negative radius, indicating that no
 * safety has been calculated. It contains no points, and neither does a sphere
 * with a safety distance of zero.
 */
struct SafetySphere
{
    Real3 center{0, 0, 0};  //!< Point where the safety was calculated [len]
    real_type radius{-1};  //!< Safety distance at the center [len]

    //! Whether the sphere has a nonzero radius
    explicit CELER_FUNCTION operator bool() const { return radius > 0; }

    //! Whether the safety distance was calculated at this exact point
    CELER_FUNCTION bool is_centered_on(Real3 const& pos) const
    {
        return radius >= 0 && center == pos;
    }

    // Whether a point is inside the sphere by at least a tolerance
    inline CELER_FUNCTION bool contains(Real3 const& pos, real_type tol) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Whether a point is inside the sphere by at least a tolerance.
 *
 * This is true if the ball of radius \c tol about \c pos is entirely inside
 * the sphere:
 * \f[
   |\mathbf{x} - \mathbf{c}| + \delta \le r \,.
 * \f]
 * An empty sphere (zero or negative radius) contains no points.
 */
CELER_FUNCTION bool SafetySphere::contains(Real3 const& pos,
                                           real_type tol) const
{
    CELER_EXPECT(tol >= 0);
    real_type const inner_radius = radius - tol;
    if (!(inner_radius > 0))
    {
        return false;
    }
    real_type dist_sq = 0;
    for (int i = 0; i < 3; ++i)
    {
        dist_sq += ipow<2>(pos[i] - center[i]);
    }
    return dist_sq <= ipow<2>(inner_radius);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
