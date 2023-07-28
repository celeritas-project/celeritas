//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/ConeAligned.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Span.hh"
#include "corecel/math/ArrayUtils.hh"
#include "orange/OrangeTypes.hh"

#include "detail/QuadraticSolver.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Axis-aligned double-sheeted cone.
 *
 * For a cone parallel to the x axis:
 * \f[
    (y - y_0)^2 + (z - z_0)^2 - t^2 (x - x_0)^2 = 0
   \f]
 */
template<Axis T>
class ConeAligned
{
  public:
    //@{
    //! \name Type aliases
    using Intersections = Array<real_type, 2>;
    using Storage = Span<const real_type, 4>;
    //@}

    //// CLASS ATTRIBUTES ////

    // Surface type identifier
    static CELER_CONSTEXPR_FUNCTION SurfaceType surface_type();

    //! Safety is intersection along surface normal
    static CELER_CONSTEXPR_FUNCTION bool simple_safety() { return false; }

  public:
    //// CONSTRUCTORS ////

    // Construct with radius
    inline CELER_FUNCTION ConeAligned(Real3 const& origin, real_type tangent);

    // Construct from raw data
    explicit inline CELER_FUNCTION ConeAligned(Storage);

    //// ACCESSORS ////

    //! Get the origin position along the normal axis
    CELER_FUNCTION Real3 const& origin() const { return origin_; }

    //! Get the square of the tangent
    CELER_FUNCTION real_type tangent_sq() const { return tsq_; }

    //! Get a view to the data for type-deleted storage
    CELER_FUNCTION Storage data() const { return {&origin_[0], 4}; }

    //// CALCULATION ////

    // Determine the sense of the position relative to this surface
    inline CELER_FUNCTION SignedSense calc_sense(Real3 const& pos) const;

    // Calculate all possible straight-line intersections with this surface
    inline CELER_FUNCTION Intersections calc_intersections(
        Real3 const& pos, Real3 const& dir, SurfaceState on_surface) const;

    // Calculate outward normal at a position
    inline CELER_FUNCTION Real3 calc_normal(Real3 const& pos) const;

  private:
    static CELER_CONSTEXPR_FUNCTION int t_index();
    static CELER_CONSTEXPR_FUNCTION int u_index();
    static CELER_CONSTEXPR_FUNCTION int v_index();

    // Location of the vanishing point
    Real3 origin_;

    // Quadric value
    real_type tsq_;
};

//---------------------------------------------------------------------------//
// TYPE ALIASES
//---------------------------------------------------------------------------//

using ConeX = ConeAligned<Axis::x>;
using ConeY = ConeAligned<Axis::y>;
using ConeZ = ConeAligned<Axis::z>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Surface type identifier.
 */
template<Axis T>
CELER_CONSTEXPR_FUNCTION SurfaceType ConeAligned<T>::surface_type()
{
    return T == Axis::x   ? SurfaceType::kx
           : T == Axis::y ? SurfaceType::ky
           : T == Axis::z ? SurfaceType::kz
                          : SurfaceType::size_;
}

//---------------------------------------------------------------------------//
/*!
 * Construct from origin and tangent of the angle of its opening.
 *
 * Given the trangular cross section of one octant of the cone, the tangent is
 * the ratio of the height to the base.
 */
template<Axis T>
CELER_FUNCTION
ConeAligned<T>::ConeAligned(Real3 const& origin, real_type tangent)
    : origin_{origin}, tsq_{ipow<2>(tangent)}
{
    CELER_EXPECT(tangent >= 0);
}

//---------------------------------------------------------------------------//
/*!
 * Construct from raw data.
 */
template<Axis T>
CELER_FUNCTION ConeAligned<T>::ConeAligned(Storage data)
    : origin_{data[0], data[1], data[2]}, tsq_{data[3]}
{
}

//---------------------------------------------------------------------------//
/*!
 * Determine the sense of the position relative to this surface.
 */
template<Axis T>
CELER_FUNCTION SignedSense ConeAligned<T>::calc_sense(Real3 const& pos) const
{
    real_type const x = pos[t_index()] - origin_[t_index()];
    real_type const y = pos[u_index()] - origin_[u_index()];
    real_type const z = pos[v_index()] - origin_[v_index()];

    return real_to_sense((-tsq_ * ipow<2>(x)) + ipow<2>(y) + ipow<2>(z));
}

//---------------------------------------------------------------------------//
/*!
 * Calculate all possible straight-line intersections with this surface.
 *
 * \f[
    (y - yc)^2 + (z - zc)^2 - t^2 * (x - xc)^2 = 0
   \f]
 */
template<Axis T>
CELER_FUNCTION auto
ConeAligned<T>::calc_intersections(Real3 const& pos,
                                   Real3 const& dir,
                                   SurfaceState on_surface) const
    -> Intersections
{
    // Expand translated positions into 'xyz' coordinate system
    real_type const x = pos[t_index()] - origin_[t_index()];
    real_type const y = pos[u_index()] - origin_[u_index()];
    real_type const z = pos[v_index()] - origin_[v_index()];

    real_type const u = dir[t_index()];
    real_type const v = dir[u_index()];
    real_type const w = dir[v_index()];

    // Scaled direction
    real_type a = (-tsq_ * ipow<2>(u)) + ipow<2>(v) + ipow<2>(w);
    real_type half_b = (-tsq_ * x * u) + (y * v) + (z * w);
    real_type c = (-tsq_ * ipow<2>(x)) + ipow<2>(y) + ipow<2>(z);

    return detail::QuadraticSolver::solve_general(a, half_b, c, on_surface);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate outward normal at a position.
 */
template<Axis T>
CELER_FUNCTION Real3 ConeAligned<T>::calc_normal(Real3 const& pos) const
{
    Real3 norm;
    for (int i = 0; i < 3; ++i)
    {
        norm[i] = pos[i] - origin_[i];
    }
    norm[t_index()] *= -tsq_;

    normalize_direction(&norm);
    return norm;
}

//---------------------------------------------------------------------------//
//!@{
//! Integer index values for primary and orthogonal axes.
template<Axis T>
CELER_CONSTEXPR_FUNCTION int ConeAligned<T>::t_index()
{
    return static_cast<int>(T);
}
template<Axis T>
CELER_CONSTEXPR_FUNCTION int ConeAligned<T>::u_index()
{
    return static_cast<int>(T == Axis::x ? Axis::y : Axis::x);
}
template<Axis T>
CELER_CONSTEXPR_FUNCTION int ConeAligned<T>::v_index()
{
    return static_cast<int>(T == Axis::z ? Axis::y : Axis::z);
}
//!@}

//---------------------------------------------------------------------------//
}  // namespace celeritas
