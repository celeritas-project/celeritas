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
#include <corecel/io/Logger.hh>  // Remove when done
#include <corecel/math/ArrayOperators.hh>
#include <corecel/math/ArrayUtils.hh>
#include <orange/OrangeTypes.hh>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * A triangle defined by three vertices.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
    Tri ...;
   \endcode
 */
template<typename T>
class Tri
{
  public:
    //!@{
    //! \name Type aliases
    using Vertex = Array<T, 3>;
    using Intersections = Array<real_type, 1>;
    //!@}

  public:
    // Construct from 3 vertices
    inline Tri(Vertex const& a, Vertex const& b, Vertex const& c);

    //! Vertex a
    CELER_FUNCTION Vertex const& a() const { return a_; }

    //! Vertex b
    CELER_FUNCTION Vertex const& b() const { return b_; }

    //! Vertex c
    CELER_FUNCTION Vertex const& c() const { return c_; }

    // Calculate possible straight-line intersections with this triangle
    inline CELER_FUNCTION Intersections calc_intersections(
        Vertex const& pos, Vertex const& dir, SurfaceState on_surface) const;

  private:
    // Calculate the centroid of the triangle
    inline CELER_FUNCTION Vertex calc_centroid() const;

    // Vertex A
    Vertex a_;
    // Vertex B
    Vertex b_;
    // Vertex C
    Vertex c_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from 3 vertices.
 */
template<typename T>
CELER_FUNCTION Tri<T>::Tri(Vertex const& a, Vertex const& b, Vertex const& c)
    : a_(a), b_(b), c_(c)
{
    // Edge 1 and Edge 2
    auto edge1 = b_ - a_;
    auto edge2 = c_ - a_;

    // Calculate the normal vector
    auto normal = cross_product(edge1, edge2);

    // Check if orientation is CCW relative to global z axis. Swap b and c
    // vertices if not.
    Vertex z_axis = {0, 0, 1};
    auto dot = dot_product(normal, z_axis);
    if (dot < 0.0)
    {
        b_ = c;
        c_ = b;
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
    Vertex const& pos, Vertex const& dir, SurfaceState on_surface) const
    -> Intersections
{
    constexpr T epsilon = std::numeric_limits<T>::epsilon();

    // Edge 1 and Edge 2
    auto edge1 = b_ - a_;
    auto edge2 = c_ - a_;

    // Calculate the normal vector
    auto normal = cross_product(edge1, edge2);
    CELER_LOG(diagnostic) << "normal: " << normal;

    // if (dot_product(dir, normal) > 0.0)
    //{
    //     CELER_LOG(diagnostic) << "dot_product: " << dot_product(dir,
    //     normal); return {no_intersection()};
    // }

    auto rayCrossEdge2 = cross_product(dir, edge2);
    T det = dot_product(edge1, rayCrossEdge2);

    // Ray is parallel to triangle
    if (std::fabs(det) < epsilon)
    {
        return {no_intersection()};
    }

    T inv_det = 1.0 / det;
    auto s = pos - a_;
    T u = inv_det * dot_product(s, rayCrossEdge2);

    // Ray passes outside edge2's bounds
    if (u < -epsilon || (u - 1) > epsilon)
    {
        return {no_intersection()};
    }

    auto sCrossEdge1 = cross_product(s, edge1);
    T v = inv_det * dot_product(dir, sCrossEdge1);

    // Ray passes outside edge1's bounds
    if (v < -epsilon || (u + v - 1) > epsilon)
    {
        return {no_intersection()};
    }

    // Compute distance to intersection
    T t = inv_det * dot_product(edge2, sCrossEdge1);

    if (t > epsilon)
    {
        return {t};
    }
    else
    {
        return {no_intersection()};
    }
}

//---------------------------------------------------------------------------//
/*!
 * Construct from 3 vertices.
 */
template<typename T>
CELER_FUNCTION auto Tri<T>::calc_centroid() const -> Vertex
{
    return (a_ + b_ + c_) / 3.0;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
