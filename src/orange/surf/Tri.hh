//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/Tri.hh
//---------------------------------------------------------------------------//
#pragma once

#include <limits>
#include <corecel/Macros.hh>
#include <corecel/cont/Array.hh>
#include <corecel/math/ArrayOperators.hh>
#include <corecel/math/ArrayUtils.hh>
#include <orange/OrangeTypes.hh>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * A triangle defined by three vertices.
 *
 * On construction, the triangle will be updated to a CCW orientation.
 *
 * The triangle cannot be degenerate with zero area:
 *   1. No two vertices can be equal,
 *   2. The three vertices are not colinear: a + b != c
 *
 */
template<typename T>
class Tri
{
  public:
    //!@{
    //! \name Type aliases
    using real_type = T;
    using Real3 = Array<real_type, 3>;
    using Vertices = Array<Real3, 3>;
    using Intersections = Array<real_type, 1>;
    //!@}

  public:
    // Construct from 3 vertices
    inline Tri(Vertices const& vertices);

    //! The vertices defining the triangle
    CELER_FUNCTION Vertices const& verts() const { return verts_; }

    // Calculate straight-line intersections with this triangle
    inline CELER_FUNCTION Intersections calc_intersections(
        Real3 const& pos, Real3 const& dir, SurfaceState on_surface) const;

  private:
    //// DATA ////

    // Vertices defining the triangle
    Vertices verts_;

    //// METHODS ////

    // Calculate the centroid of the triangle
    inline CELER_FUNCTION Real3 calc_centroid() const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from 3 vertices.
 */
template<typename T>
CELER_FUNCTION Tri<T>::Tri(Vertices const& verts) : verts_(verts)
{
    // Assert that triangle is not degenerate
    CELER_ASSERT(verts[0] != verts[1]);
    CELER_ASSERT(verts[0] != verts[2]);
    CELER_ASSERT(verts[1] != verts[2]);
    CELER_ASSERT(verts[0] + verts[1] != verts[2]);

    // Edge 1 and Edge 2
    auto edge1 = verts_[1] - verts[0];
    auto edge2 = verts_[2] - verts[0];

    // Check if orientation is CCW relative to global z axis. Swap 2nd and
    // third vertices if not
    if (edge1[0] * edge2[1] - edge1[1] * edge2[0] < 0.0)
    {
        verts_[1] = verts[2];
        verts_[2] = verts[1];
    }
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the intersection point.
 *
 * This is an implementation of the Moller-Trumbore intersection algorithm
 * https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
 */
template<typename T>
CELER_FUNCTION auto Tri<T>::calc_intersections(
    Real3 const& pos, Real3 const& dir, SurfaceState on_surface) const
    -> Intersections
{
    constexpr real_type epsilon = std::numeric_limits<real_type>::epsilon();

    // Edge 1 and Edge 2
    auto edge1 = verts_[1] - verts_[0];
    auto edge2 = verts_[2] - verts_[0];

    // Calculate the normal vector
    auto normal = cross_product(edge1, edge2);

    auto ray_cross_edge2 = cross_product(dir, edge2);
    real_type det = dot_product(edge1, ray_cross_edge2);

    // Ray is parallel to triangle
    if (std::fabs(det) < epsilon)
    {
        return {real_type(no_intersection())};
    }

    real_type inv_det = 1.0 / det;
    auto s = pos - verts_[0];
    real_type u = inv_det * dot_product(s, ray_cross_edge2);

    // Ray intersects edge2 outside the triangle
    if (u < epsilon || (u - 1) > epsilon)
    {
        return {real_type(no_intersection())};
    }

    auto s_cross_edge1 = cross_product(s, edge1);
    real_type v = inv_det * dot_product(dir, s_cross_edge1);

    // Ray intersects edge1 outside the triangle
    if (v < epsilon || (u + v - 1) > epsilon)
    {
        return {real_type(no_intersection())};
    }

    // Compute distance to intersection
    real_type t = inv_det * dot_product(edge2, s_cross_edge1);

    if (t > epsilon)
    {
        return {t};
    }
    else
    {
        return {real_type(no_intersection())};
    }
}

//---------------------------------------------------------------------------//
/*!
 * Construct from 3 vertices.
 */
template<typename T>
CELER_FUNCTION auto Tri<T>::calc_centroid() const -> Real3
{
    return (verts_[0] + verts_[1] + verts_[2]) / 3.0;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
