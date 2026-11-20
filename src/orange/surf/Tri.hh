//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/Tri.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Span.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/math/SoftEqual.hh"
#include "orange/OrangeTypes.hh"

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
 *   2. The three vertices are not colinear: a + b != c, where a, b, and c are
 * the triangle side lengths.
 *
 * The ray-triangle intersection utilizes the Moller-Trumbore algorithm.
 * https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
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
    using Edges = Array<Real3, 2>;
    using StorageSpan = Span<real_type const, 3>;
    //!@}

  public:
    // Construct from 3 vertices
    inline explicit CELER_FUNCTION Tri(Vertices const& verts);

    // Construct from raw data
    template<class R>
    explicit inline CELER_FUNCTION Tri(Span<R, StorageSpan::extent>);

    //! The base vertex of the triangle
    inline CELER_FUNCTION Real3 const& vertex() const { return vertex_; }

    //! The two edges connected to the triangle
    inline CELER_FUNCTION Edges const& edges() const { return edges_; }

    // Calculate straight-line intersections with this triangle
    inline CELER_FUNCTION real_type calc_intersections(
        Real3 const& pos, Real3 const& dir, SurfaceState on_surface) const;

  private:
    //// DATA ////

    // The first vertex of the triangle
    Real3 vertex_;
    // Two edges of the triangle connected to vertex_
    Edges edges_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from 3 vertices.
 */
template<typename T>
CELER_FUNCTION Tri<T>::Tri(Vertices const& verts) : vertex_(verts[0])
{
    // Store the two edges
    edges_[0] = verts[1] - verts[0];
    edges_[1] = verts[2] - verts[0];

    // Check that triangle is not degenerate
    CELER_EXPECT([&] {
        return !soft_zero(norm(cross_product(edges_[0], edges_[1])));
    }());

    // Check if orientation is CCW relative to global z axis. Swap 2nd and
    // third vertices if not
    if (edges_[0][0] * edges_[1][1] - edges_[0][1] * edges_[1][0]
        < real_type{0})
    {
        edges_[0] = verts[2] - verts[0];
        edges_[1] = verts[1] - verts[0];
    }
}

//---------------------------------------------------------------------------//
/*!
 * Construct from raw data.
 */
template<typename T>
template<class R>
CELER_FUNCTION Tri<T>::Tri(Span<R, StorageSpan::extent> data)
    : vertex_{data[0]}, edges_{data[1], data[2]}
{
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the intersection point.
 *
 * This is an implementation of the Moller-Trumbore intersection algorithm.
 */
template<typename T>
CELER_FUNCTION auto Tri<T>::calc_intersections(
    Real3 const& pos, Real3 const& dir, SurfaceState on_surface) const
    -> real_type
{
    // Early return if particle is on surface to prevent re-entry
    // TODO: Revisit as triangle mesh tracker progresses
    if (on_surface == SurfaceState::on)
    {
        return real_type(no_intersection());
    }

    auto ray_cross_edge2 = cross_product(dir, edges_[1]);
    real_type det = dot_product(edges_[0], ray_cross_edge2);

    // Ray is parallel to triangle
    if (soft_zero(det))
    {
        return real_type(no_intersection());
    }

    real_type inv_det = 1 / det;
    auto s = pos - vertex_;
    real_type u = inv_det * dot_product(s, ray_cross_edge2);

    // Ray intersects edge2 outside the triangle
    if (u < real_type{0} || u > real_type{1})
    {
        return real_type(no_intersection());
    }

    auto s_cross_edge1 = cross_product(s, edges_[0]);
    real_type v = inv_det * dot_product(dir, s_cross_edge1);

    // Ray intersects edge1 outside the triangle
    if (v < real_type{0} || u + v > real_type{1})
    {
        return real_type(no_intersection());
    }

    // Compute distance to intersection
    real_type t = inv_det * dot_product(edges_[1], s_cross_edge1);

    if (t > real_type{0})
    {
        return t;
    }
    else
    {
        return real_type(no_intersection());
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
