//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/Toroid.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/cont/Array.hh"
#include "corecel/cont/Span.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/math/FerrariSolver.hh"
#include "orange/OrangeTypes.hh"
#include "orange/SenseUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Z-aligned Elliptical Toroid.
 *
 * An elliptical toroid is a shape created by revolving an axis-aligned ellipse
 * around a central axis. This shape can be used in everything from pipe bends
 * to tokamaks in fusion reactors. It possesses a major radius \em r, and
 * ellipse radii \em a and \em b, as shown in the below diagram:
 *
 * \verbatim
       ___   _________   ___
     /  |  \           /     \
    /   b   \         /       \
   |    |    |       |         |
   |-a--+    |   o-----r--+    |
   |         |       |         |
    \       /         \       /
     \...../...........\...../

   \endverbatim
 *
 * This torus can be defined with the following quartic equation:
 * \f[
 *   (x^2 + y^2 + p*y^2 + B_0) - A_0 * (x^2 + y^2) = 0
 * \f]
 * where \f$p = a^2/b^2 \f$,  \f$A_0 = 4 r^2 \f$, and \f$ B_0 = (r^2-a^2)\f$.
 *
 * The intersection is calculated using an algorithm from
 * \cite{arvo-graphicsgems-1995}.
 */
class Toroid
{
  public:
    //@{
    //! \name Type aliases
    using Intersections = Array<real_type, 4>;
    using StorageSpan = Span<real_type const, 6>;
    using Real3 = Array<real_type, 3>;
    using Real4 = Array<real_type, 4>;
    using Real5 = Array<real_type, 5>;
    //@}

  public:
    //// CLASS ATTRIBUTES ////

    // Surface type identifier
    static SurfaceType surface_type()
    {
        CELER_NOT_IMPLEMENTED("runtime toroid");
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
    CELER_FUNCTION StorageSpan data() const
    {
        static_assert(sizeof(*this) == sizeof(real_type) * 6);
        return {&origin_[0], 6};
    }

    //// CALCULATION ///

    // Determine the sense of the position relative to this surface
    inline CELER_FUNCTION SignedSense calc_sense(Real3 const& pos) const;

    // Calculate all possible straight-line intersections with this surface
    inline CELER_FUNCTION Intersections calc_intersections(
        Real3 const& pos, Real3 const& dir, SurfaceState on_surface) const;

    // Calculate outward normal at a position
    inline CELER_FUNCTION Real3 calc_normal(Real3 const& pos) const;

  private:
    //// DATA ////

    //! Location of center of toroid
    Real3 origin_;

    // Radii
    //! Radius from origin to center of revolved ellipse (along xy plane)
    real_type r_;
    //! Horizontal radius of revolved ellipse (along xy plane)
    real_type a_;
    //! Vertical radius of revolved ellipse (along z axis)
    real_type b_;

    //// HELPER FUNCTIONS ////

    // Calculate the coefficients of the polynomial for ray intersection
    inline CELER_FUNCTION Real5 calc_intersection_polynomial(
        Real3 const& pos, Real3 const& dir, SurfaceState on_surface) const;

    // Shorthand to square a number
    static CELER_CONSTEXPR_FUNCTION real_type sq(real_type val);
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
/*!
 * Determine the sense of the position relative to this surface.
 *
 * For a toroid, being inside the toroid (i) counts as inside, outside
 * (including in the 'hole' region) (o) as outside, and on the surface exactly
 * as on (s).
 * \verbatim
       ___   _________   ___
     /     \           /     \
    /       \     o   /       \
   |         |       |         | o
   |         |       |    i    s
    \       /         \       /
     \     /           \     /
       ⁻⁻⁻   ⁻⁻⁻⁻⁻⁻⁻⁻⁻   ⁻⁻⁻
   \endverbatim
 */
CELER_FUNCTION SignedSense Toroid::calc_sense(Real3 const& pos) const
{
    auto [x0, y0, z0] = pos - origin_;

    real_type val = (sq(sq(x0) + sq(y0) + sq(z0 * a_ / b_) + (sq(r_) - sq(a_)))
                     - (4 * sq(r_)) * (sq(x0) + sq(y0)));

    return real_to_sense(val);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate all possible straight-line intersections between the given ray and
 * this surface.
 */
CELER_FUNCTION auto Toroid::calc_intersections(Real3 const& pos,
                                               Real3 const& dir,
                                               SurfaceState on_surface) const
    -> Intersections
{
    Real5 abcde = calc_intersection_polynomial(pos, dir, on_surface);
    FerrariSolver solve{};  // Default tolerance
    Intersections roots;

    if (on_surface == SurfaceState::on)
    {
        auto [a, b, c, d, e] = abcde;
        roots = solve(Real4{a, b, c, d});
    }
    else
    {
        roots = solve(abcde);
    }

    return roots;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the coefficients of the polynomial corresponding to the given
 * ray's intersections with the toroid.
 *
 * Written referencing Graphics Gems II\cite{arvo-graphicsgems-1995}.
 */
CELER_FUNCTION auto
Toroid::calc_intersection_polynomial(Real3 const& pos,
                                     Real3 const& dir,
                                     SurfaceState on_surface) const -> Real5
{
    auto [x0, y0, z0] = pos - origin_;
    auto [ax, ay, az] = make_unit_vector(dir);

    // Intermediate terms
    real_type p = sq(a_) / sq(b_);

    real_type f = 1 - sq(az);
    real_type g = f + p * sq(az);
    real_type h = 2 * (x0 * ax + y0 * ay);
    real_type t = sq(x0) + sq(y0);
    real_type q = 4 * sq(r_) / sq(g);
    real_type m = (h + 2 * p * z0 * az) / g;
    real_type u = (t + p * sq(z0) + sq(r_) - sq(a_)) / g;

    // Polynomial coefficients, i.e. cn*x^n
    real_type c4 = 1;
    real_type c3 = 2 * m;
    real_type c2 = sq(m) + 2 * u - q * f;
    real_type c1 = 2 * m * u - q * h;
    real_type c0 = on_surface == SurfaceState::on ? 0 : sq(u) - q * t;
    // Potential refinement of c0 if close to 0?

    return Real5{c4, c3, c2, c1, c0};
}

//---------------------------------------------------------------------------//
/*!
 * Calculate outward facing normal at a position on or close to the surface.
 */
CELER_FUNCTION auto Toroid::calc_normal(Real3 const& pos) const -> Real3
{
    auto [x0, y0, z0] = pos - origin_;

    real_type d = hypot(x0, y0);
    real_type f = 2 * (d - r_) / (d * sq(a_));
    Real3 n{x0 * f, y0 * f, (2 * z0) / (sq(b_))};
    return make_unit_vector(n);
}

//---------------------------------------------------------------------------//
/*!
 * Shorthand for power macro for readability, as squares are used frequently
 */
CELER_CONSTEXPR_FUNCTION real_type Toroid::sq(real_type val)
{
    return ipow<2>(val);
}

}  // namespace celeritas
