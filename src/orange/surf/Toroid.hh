//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/Toroid.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iostream>

#include "corecel/cont/Array.hh"
#include "corecel/cont/Span.hh"
#include "corecel/math/Algorithms.hh"
#include "orange/OrangeTypes.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Z-aligned Elliptical Toroid.
 *
 * An elliptical toroid is a shape created by revolving an axis-aligned ellipse
 * around a central axis. This shape can be used in everything from pipe bends
 * to tokamaks in fusion reactors. Possesses a major radius r, and ellipse
 * radii a and b, as shown in the below diagram:
 *     ___   _________   ___
 *   /  |  \           /     \
 *  /   b   \         /       \
 * |    |    |       |         |
 * |-a--+    |   o-----r--+    |
 * |         |       |         |
 *  \       /         \       /
 *   \     /           \     /
 *     ⁻⁻⁻   ⁻⁻⁻⁻⁻⁻⁻⁻⁻   ⁻⁻⁻
 *
 * This torus can be defined with the following quartic equation:
 * \f[
 *   (x^2 + y^2 + p*y^2 + B_0) - A_0 * (x^2 + y^2) = 0
 * \f]
 * where \f[p = a^2/b^2, A_0 = 4*r^2, and B_0 = (r^2-a^2)\f].
 */
class Toroid
{
  public:
    //@{
    //! \name Type aliases
    using Intersections = Array<real_type, 4>;
    using StorageSpan = Span<real_type const, 4>;
    using Real3 = Array<real_type, 3>;
    //@}

  public:
    //// CLASS ATTRIBUTES ////

    // Surface type identifier
    static CELER_CONSTEXPR_FUNCTION SurfaceType surface_type()
    {
        return SurfaceType::s;
    }

    //! Safety distance is calculable w/xy of normal and ellipse safety
    //! distance, but this is out of scope at first and might not be trivially
    //! calculable
    //! https://web.archive.org/web/20170829172516/https://www.spaceroots.org/documents/distance/distance-to-ellipse.pdf
    static CELER_CONSTEXPR_FUNCTION bool simple_safety() { return false; }

  public:
    //// CONSTRUCTORS ////

    explicit Toroid(Real3 const& origin,
                    real_type major_radius,
                    real_type ellipse_xy_radius,
                    real_type ellipse_z_radius);

    // Construct from raw data
    template<class R>
    explicit inline CELER_FUNCTION Toroid(Span<R, StorageSpan::extent>);

    //// ACCESSORS ////

    //! Center of the toroid (in the donut hole)
    CELER_FUNCTION Real3 const& origin() const { return origin_; }

    //! Radius from origin to center of revolved ellipse
    CELER_FUNCTION real_type major_radius() const { return r_; }

    //! Radius of revolved ellipse along xy plane
    CELER_FUNCTION real_type ellipse_xy_radius() const { return a_; }

    //! Radius of revolved ellipse along z axis
    CELER_FUNCTION real_type ellipse_z_radius() const { return b_; }

    //! View of data for type-deleted storage
    CELER_FUNCTION StorageSpan data() const { return {&origin_[0], 6}; }

    //// CALCULATION ///

    // Determine the sense of the position relative to this surface
    inline CELER_FUNCTION SignedSense calc_sense(Real3 const& pos) const;

    // Calculate all possible straight-line intersections with this surface
    inline CELER_FUNCTION Intersections calc_intersections(
        Real3 const& pos, Real3 const& dir, SurfaceState on_surface) const;

    // Calculate outward normal at a position
    inline CELER_FUNCTION Real3 calc_normal(Real3 const& pos) const;

  private:
    // Location of center of toroid
    Real3 origin_;

    // Radii
    real_type r_;  // Radius from origin to center of revolved ellipse (along
                   // xy plane)
    real_type a_;  // Horizontal radius of revolved ellipse (along xy plane)
    real_type b_;  // Vertical radius of revolved ellipse (along z axis)
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from raw data.
 */
template<class R>
CELER_FUNCTION Toroid::Toroid(Span<R, StorageSpan::extent> data)
    : origin_{data[0], data[1], data[2]}, r_{data[3]}, a_{data[4]}, b_{data[5]}
{
}

//---------------------------------------------------------------------------//
/**
 * Determine the sense of the position relative to this surface.
 *
 * For a toroid, being inside the toroid (i) counts as inside, outside
 * (including in the 'hole' region) (o) as outside, and on the surface exactly
 * as on (s).
 *     ___   _________   ___
 *   /     \           /     \
 *  /       \     o   /       \
 * |         |       |         | o
 * |         |       |    i    s
 *  \       /         \       /
 *   \     /           \     /
 *     ⁻⁻⁻   ⁻⁻⁻⁻⁻⁻⁻⁻⁻   ⁻⁻⁻
 */
CELER_FUNCTION SignedSense Toroid::calc_sense(Real3 const& pos) const
{
    auto [x, y, z] = pos;
    real_type x0 = x - origin_[0];
    real_type y0 = y - origin_[1];
    real_type z0 = z - origin_[2];

    real_type val = (ipow<2>(ipow<2>(x0) + ipow<2>(y0) + ipow<2>(z0 * a_ / b_)
                             + (ipow<2>(r_) - ipow<2>(a_)))
                     - (4 * ipow<2>(r_)) * (ipow<2>(x0) + ipow<2>(y0)));
    if (val < 0)
        return SignedSense::inside;
    else if (val > 0)
        return SignedSense::outside;
    else
        return SignedSense::on;
}

//---------------------------------------------------------------------------//
/**
 * Calculate all possible straight-line intersections between the given ray and
 * this surface.
 */
CELER_FUNCTION auto Toroid::calc_intersections(Real3 const& pos,
                                               Real3 const& dir,
                                               SurfaceState on_surface) const
    -> Intersections
{
    return Intersections{no_intersection(),
                         no_intersection(),
                         no_intersection(),
                         no_intersection()};
}

//---------------------------------------------------------------------------//
/**
 * Calculate outward facing normal at a position on or close to the surface.
 */
CELER_FUNCTION auto Toroid::calc_normal(Real3 const& pos) const -> Real3
{
    return Real3{0, 0, 0};
}

}  // namespace celeritas
