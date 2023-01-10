//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/CylCentered.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Span.hh"
#include "corecel/math/ArrayUtils.hh"
#include "orange/OrangeTypes.hh"

#include "detail/QuadraticSolver.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Axis-aligned cylinder centered about the origin.
 *
 * The cylinder is centered along an Axis template parameter.
 *
 * For a cylinder along the x axis:
 * \f[
    y^2 + z^2 - R^2 = 0
   \f]
 *
 * This is an optimization of the Cyl. The motivations are:
 * - Many geometries have units with concentric cylinders centered about the
 *   origin, so having this as a special case reduces the memory usage of those
 *   units (improved memory localization).
 * - Cylindrical mesh geometries have lots of these cylinders, so efficient
 *   tracking through its cells should make this optimization worthwhile.
 */
template<Axis T>
class CylCentered
{
  public:
    //@{
    //! Type aliases
    using Intersections = Array<real_type, 2>;
    using Storage = Span<const real_type, 1>;
    //@}

    //// CLASS ATTRIBUTES ////

    // Surface type identifier
    static CELER_CONSTEXPR_FUNCTION SurfaceType surface_type();

    //! Safety is intersection along surface normal
    static CELER_CONSTEXPR_FUNCTION bool simple_safety() { return true; }

  public:
    //// CONSTRUCTORS ////

    // Construct with radius
    explicit inline CELER_FUNCTION CylCentered(real_type radius);

    // Construct from raw data
    explicit inline CELER_FUNCTION CylCentered(Storage);

    //// ACCESSORS ////

    //! Get the square of the radius
    CELER_FUNCTION real_type radius_sq() const { return radius_sq_; }

    //! Get a view to the data for type-deleted storage
    CELER_FUNCTION Storage data() const { return {&radius_sq_, 1}; }

    //// CALCULATION ////

    // Determine the sense of the position relative to this surface
    inline CELER_FUNCTION SignedSense calc_sense(Real3 const& pos) const;

    // Calculate all possible straight-line intersections with this surface
    inline CELER_FUNCTION Intersections calc_intersections(
        Real3 const& pos, Real3 const& dir, SurfaceState on_surface) const;

    // Calculate outward normal at a position
    inline CELER_FUNCTION Real3 calc_normal(Real3 const& pos) const;

  private:
    //! Square of cylinder radius
    real_type radius_sq_;

    static CELER_CONSTEXPR_FUNCTION int t_index();
    static CELER_CONSTEXPR_FUNCTION int u_index();
    static CELER_CONSTEXPR_FUNCTION int v_index();
};

//---------------------------------------------------------------------------//
// TYPE ALIASES
//---------------------------------------------------------------------------//

using CCylX = CylCentered<Axis::x>;
using CCylY = CylCentered<Axis::y>;
using CCylZ = CylCentered<Axis::z>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Surface type identifier.
 */
template<Axis T>
CELER_CONSTEXPR_FUNCTION SurfaceType CylCentered<T>::surface_type()
{
    return (T == Axis::x
                ? SurfaceType::cxc
                : (T == Axis::y ? SurfaceType::cyc : SurfaceType::czc));
}

//---------------------------------------------------------------------------//
/*!
 * Construct with radius.
 */
template<Axis T>
CELER_FUNCTION CylCentered<T>::CylCentered(real_type radius)
    : radius_sq_(ipow<2>(radius))
{
    CELER_EXPECT(radius > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Construct from raw data.
 */
template<Axis T>
CELER_FUNCTION CylCentered<T>::CylCentered(Storage data) : radius_sq_(data[0])
{
}

//---------------------------------------------------------------------------//
/*!
 * Determine the sense of the position relative to this surface.
 */
template<Axis T>
CELER_FUNCTION SignedSense CylCentered<T>::calc_sense(Real3 const& pos) const
{
    const real_type u = pos[u_index()];
    const real_type v = pos[v_index()];

    return real_to_sense(ipow<2>(u) + ipow<2>(v) - radius_sq_);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate all possible straight-line intersections with this surface.
 */
template<Axis T>
CELER_FUNCTION auto
CylCentered<T>::calc_intersections(Real3 const& pos,
                                   Real3 const& dir,
                                   SurfaceState on_surface) const
    -> Intersections
{
    // 1 - \omega \dot e
    const real_type a = 1 - ipow<2>(dir[t_index()]);

    if (a != 0)
    {
        const real_type u = pos[u_index()];
        const real_type v = pos[v_index()];

        // b/2 = \omega \dot (x - x_0)
        detail::QuadraticSolver solve_quadric(
            a, dir[u_index()] * u + dir[v_index()] * v);
        if (on_surface == SurfaceState::off)
        {
            // c = (x - x_0) \dot (x - x_0) - R * R
            return solve_quadric(ipow<2>(u) + ipow<2>(v) - radius_sq_);
        }
        else
        {
            // Solve degenerate case (c=0)
            return solve_quadric();
        }
    }
    else
    {
        // No intersection if we're traveling along the cylinder axis
        return {no_intersection(), no_intersection()};
    }
}

//---------------------------------------------------------------------------//
/*!
 * Calculate outward normal at a position.
 */
template<Axis T>
CELER_FUNCTION Real3 CylCentered<T>::calc_normal(Real3 const& pos) const
{
    Real3 norm{0, 0, 0};

    norm[u_index()] = pos[u_index()];
    norm[v_index()] = pos[v_index()];

    normalize_direction(&norm);
    return norm;
}

//---------------------------------------------------------------------------//
//!@{
//! Integer index values for primary and orthogonal axes.
template<Axis T>
CELER_CONSTEXPR_FUNCTION int CylCentered<T>::t_index()
{
    return static_cast<int>(T);
}
template<Axis T>
CELER_CONSTEXPR_FUNCTION int CylCentered<T>::u_index()
{
    return static_cast<int>(T == Axis::x ? Axis::y : Axis::x);
}
template<Axis T>
CELER_CONSTEXPR_FUNCTION int CylCentered<T>::v_index()
{
    return static_cast<int>(T == Axis::z ? Axis::y : Axis::z);
}
//!@}

//---------------------------------------------------------------------------//
}  // namespace celeritas
