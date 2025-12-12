//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/Toroid.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/Array.hh"
#include "corecel/cont/Span.hh"
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
    using Intersections = Array<real_type, 3>;
    using StorageSpan = Span<real_type const, 6>;
    using Real3 = Array<real_type, 3>;
    //@}

  public:
    //// CLASS ATTRIBUTES ////

    // Surface type identifier
    static CELER_CONSTEXPR_FUNCTION SurfaceType surface_type()
    {
        return SurfaceType::tor;
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
}  // namespace celeritas
