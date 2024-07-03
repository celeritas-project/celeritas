//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/Involute.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Constants.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Span.hh"
#include "corecel/math/ArrayUtils.hh"
#include "orange/OrangeTypes.hh"

#include "detail/InvoluteSolver.hh"

namespace celeritas
{
using constants::pi;
//---------------------------------------------------------------------------//
/*!
 * Involute:
 *
 * \f[
    x = r_b * (cos(t+a) + tsin(t+a)) + x_0
    y = r_b * (sin(t+a) - tcos(t+a)) + y_0
   \f]

 * where \em t is the normal angle of the tangent to the circle of involute
 with
 * radius \em r_b from a starting angle of \em a (\f$r/h\f$ for a finite cone.

high level what it does
 */
class Involute
{
  public:
    //@{
    //! \name Type aliases
    using Intersections = Array<real_type, 3>;
    using StorageSpan = Span<real_type const, 8>;
    //@}

  public:
    //// CLASS ATTRIBUTES ////

    // Surface type identifier
    static CELER_CONSTEXPR_FUNCTION SurfaceType surface_type()
    {
        return SurfaceType::inv;
    }

    // Safety
    static CELER_CONSTEXPR_FUNCTION bool simple_safety() { return false; }

  public:
    //// CONSTRUCTORS ////
    explicit Involute(Real3 const& origin,
                      real_type radius,
                      real_type a,
                      real_type sign,
                      real_type tmin,
                      real_type tmax);

    // Construct from raw data
    template<class R>
    explicit inline CELER_FUNCTION Involute(Span<R, StorageSpan::extent>);

    //// ACCESSORS ////

    //! Get the origin position
    CELER_FUNCTION Real3 const& origin() const { return origin_; }

    //! Get involute parameters
    CELER_FUNCTION real_type r_b() const { return r_b_; }
    CELER_FUNCTION real_type a() const { return a_; }
    CELER_FUNCTION real_type sign() const { return sign_; }

    //! Get bounds of the involute
    CELER_FUNCTION real_type tmin() const { return tmin_; }
    CELER_FUNCTION real_type tmax() const { return tmax_; }

    //! Get a view to the data for type-deleted storage
    CELER_FUNCTION StorageSpan data() const { return {&origin_[0], 8}; }

    //// CALCULATION ////

    // Determine the sense of the position relative to this surface
    inline CELER_FUNCTION SignedSense calc_sense(Real3 const& pos) const;

    // Calculate all possible straight-line intersections with this surface
    inline CELER_FUNCTION Intersections calc_intersections(
        Real3 const& pos, Real3 const& dir, SurfaceState on_surface) const;

    // Calculate outward normal at a position
    inline CELER_FUNCTION Real3 calc_normal(Real3 const& pos) const;

  private:
    // Location of the center of circle of involute
    Real3 origin_;

    // Involute parameters
    real_type r_b_;
    real_type a_;
    real_type sign_;

    // Bounds
    real_type tmin_;
    real_type tmax_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
/*!
 * Construct from raw data.
 */
template<class R>
CELER_FUNCTION Involute::Involute(Span<R, StorageSpan::extent> data)
    : origin_{data[0], data[1], data[2]}
    , r_b_{data[3]}
    , a_{data[4]}
    , sign_{data[5]}
    , tmin_{data[6]}
    , tmax_{data[7]}
{
}

//---------------------------------------------------------------------------//
/*!
 * Determine the sense of the position relative to this surface.
 * This is performed by first checking if the particle is outside the bounding
 * circles given by tmin and tmax.
 *
 * Then position is compared to a point on the involute at the same radius from
 * the origin.
 *
 * Next the point is determined to be inside if an involute that passes it,
 * has an \em t that is within the bounds and an \em a greater than the
 * involute being tested for. To do this, the tangent to the involute must be
 * determined by calculating the intercepts of the circle of involute and
 * circle centered at the midpoint of the origin and the point being tested for
 * that extends to the origin. After which the angle of tangent point can be
 * determined and used to calculate \em a via: /f[ a = angle - t_point /f]
 *
 * To calculate the intercept of two circles it is assumed that the two circles
 * are \em xp axis. The coordinate along \em xp originating from the center of
 * one of the circles is given by: /f[
 * xp = (d^2 - r^2 + R^2)/(2d)
 * /f]
 * Where \em d is the distance between the center of the two circles, \em r is
 * the radius of the displaced circle, and \em R is the radius of the circle at
 * the orgin. Then the point \em yp can be obatined from: \f[
 * yp = +- sqrt(R^2 - xp^2)
 * \f]
 * Then to convert ( \em xp , \em yp ) to ( \em x, \em y ) the following
 * roation is applied:/f[ x = xp*cos(theta) - yp*sin(theta) y = yp*cos(theta) +
 * xp*sin(theta) /f] Where \em theta is the angle between the \em xp axis and
 * the \em x axis.
 */
CELER_FUNCTION SignedSense Involute::calc_sense(Real3 const& pos) const
{
    real_type const x = pos[0] - origin_[0];
    real_type const y = pos[1] - origin_[1];

    /*
     * Calculate distance to origin and obtain t value for distance.
     */
    real_type const rxy2 = ipow<2>(x) + ipow<2>(y);
    real_type const t_point2 = (rxy2 / ipow<2>(r_b_)) - 1;
    real_type const t_point = std::sqrt(t_point2);

    /*
     * Check if Point is on involute.
     */
    real_type const angle = t_point + a_;
    real_type const x_inv = r_b_
                            * (std::cos(angle) + t_point * std::sin(angle));
    real_type const y_inv = r_b_
                            * (std::sin(angle) - t_point * std::cos(angle));

    if (x == x_inv && y == y_inv)
    {
        return SignedSense::on;
    }

    /*
     * Check if point is in defined bounds.
     */
    if (std::fabs(t_point2) < ipow<2>(tmin_))
    {
        return SignedSense::outside;
    }
    if (std::fabs(t_point2) > ipow<2>(tmax_))
    {
        return SignedSense::outside;
    }

    /*
     * Check if point is in interval
     */

    // Describ

    // Calculate tangents
    real_type r = std::sqrt(ipow<2>(x) + ipow<2>(y));
    real_type xp = (ipow<2>(r_b_)) / (2 * r);
    real_type yp = std::sqrt(ipow<2>(r_b_) - ipow<2>(xp)) * sign_;
    real_type theta = std::atan(y / x);
    if (x < 0)
    {
        theta += pi;
    }

    Array<real_type, 2> point;
    point[0] = xp * std::cos(theta) - yp * std::sin(theta);
    point[1] = yp * std::cos(theta) + xp * std::cos(theta);

    // Calculate angle of tangent
    theta = std::acos(point[0]
                      / std::sqrt(point[0] * point[0] + point[1] * point[1]));
    if (point[1] < 0)
    {
        theta = (pi - theta) + pi;
    }
    real_type a1 = theta - t_point;
    if (std::fabs(theta) < std::fabs(tmax_ + a_) && a1 >= a_)
    {
        return SignedSense::inside;
    }

    while (std::fabs(theta) < (std::fabs(tmax_ + a_)))
    {
        theta += pi * 2 * sign_;
        a1 = theta - t_point;
        if (std::fabs(theta) < std::fabs(tmax_ + a_) && a1 >= a_)
        {
            return SignedSense::inside;
        }
    }

    return SignedSense::outside;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate all possible straight-line intersections with this surface.
 */
CELER_FUNCTION auto
Involute::calc_intersections(Real3 const& pos,
                             Real3 const& dir,
                             SurfaceState on_surface) const -> Intersections
{
    // Expand translated positions into 'xyz' coordinate system
    real_type const x = pos[0] - origin_[0];
    real_type const y = pos[1] - origin_[1];
    real_type const z = pos[2] - origin_[2];

    real_type const u = dir[0];
    real_type const v = dir[1];
    real_type const w = dir[2];

    detail::InvoluteSolver solve(r_b_, a_, sign_, tmin_, tmax_);

    return solve(Real3{x, y, z}, Real3{u, v, w});
}

//---------------------------------------------------------------------------//
/*!
 * Calculate outward normal at a position on the surface.
 * This is done by taking the derivative of the involute equation : /f[
 * normal_vec = {sin(t+a) -cos(t+a), 0}
 * /f]
 */
CELER_FORCEINLINE_FUNCTION Real3 Involute::calc_normal(Real3 const& pos) const
{
    real_type const x = pos[0] - origin_[0];
    real_type const y = pos[1] - origin_[1];

    /*
     * Calculate distance to origin and obtain t value for distance.
     */
    real_type const rxy2 = ipow<2>(x) + ipow<2>(y);
    real_type const t_point = std::sqrt((rxy2 / (ipow<2>(r_b_))) - 1) * sign_;

    /*
     * Calculate normal
     */
    real_type const angle = t_point + a_;
    Real3 normal_ = {std::sin(angle), -std::cos(angle), 0};

    return normal_;
}
}  // namespace celeritas