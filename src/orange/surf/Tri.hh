//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/Tri.hh
//---------------------------------------------------------------------------//
#pragma once

#include <corecel/Macros.hh>
#include <corecel/cont/Array.hh>
#include <corecel/math/ArrayUtils.hh>

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
    //!@}

  public:
    // Construct from 3 vertices
    inline Tri(Vertex const& a, Vertex const& b, Vertex const& c);

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
    auto edge1 = a_ - b_;
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
 * Construct from 3 vertices.
 */
template<typename T>
CELER_FUNCTION auto Tri<T>::calc_centroid() const -> Vertex
{
    return (a_ + b_ + c_) / 3.0;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
