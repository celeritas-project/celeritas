//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/Involute.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cassert>
#include <cmath>

#include "corecel/Constants.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Span.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArrayUtils.hh"
#include "orange/OrangeTypes.hh"

#include "detail/InvolutePoint.hh"
#include "detail/InvoluteSolver.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Z-aligned circular involute.
 *
 * An involute is a curve created by unwinding a chord from a shape.
 * The involute implemented here is constructed from a circle and can be made
 * to be clockwise (negative) or anti-clockwise (positive).
 * This involute is the same type of that found in HFIR, and is necessary for
 * building accurate models.
 *
 * While the parameters define the curve itself have no restriction, except for
 * the radius of involute being positive, the involute needs to be bounded for
 * there to be finite solutions to the roots of the intersection points.
 * Additionally this bound cannot exceed an interval of size of 2 to be able to
 * determine if whether a particle is in, out, or on the involute surface.
 * Lastly the involute geometry is fixed to be z-aligned.
 *
 * \f[
 *   x = r_b (\cos(t+a) + t \sin(t+a)) + x_0
 * \f]
 * \f[
 *   y = r_b (\sin(t+a) - t \cos(t+a)) + y_0
 * \f]
 *
 * where \em t is the normal angle of the tangent to the circle of involute
 * with radius \em r_b from a starting angle of \em a (\f$r/h\f$ for a finite
 * cone.
 */
class Involute
{
  public:
    //@{
    //! \name Type aliases
    using Intersections = Array<real_type, 3>;
    using StorageSpan = Span<real_type const, 6>;
    using Real2 = Array<real_type, 2>;
    //@}

  public:
    //// CLASS ATTRIBUTES ////

    // Surface type identifier
    static CELER_CONSTEXPR_FUNCTION SurfaceType surface_type()

    {
        return SurfaceType::inv;
    }

    //! Safety cannot be trivially calculated
    static CELER_CONSTEXPR_FUNCTION bool simple_safety() { return false; }

  public:
    //// CONSTRUCTORS ////

    explicit Involute(Real2 const& origin,
                      real_type radius,
                      real_type displacement,
                      Chirality sign,
                      real_type tmin,
                      real_type tmax);

    // Construct from raw data
    template<class R>
    explicit inline CELER_FUNCTION Involute(Span<R, StorageSpan::extent>);

    //// ACCESSORS ////

    //! X-Y center of the circular base of the involute
    CELER_FUNCTION Real2 const& origin() const { return origin_; }

    //! Involute circle's radius
    CELER_FUNCTION real_type r_b() const { return std::fabs(r_b_); }

    //! Displacement angle
    CELER_FUNCTION real_type displacement_angle() const { return a_; }

    // Orientation of the involute curve
    inline CELER_FUNCTION Chirality sign() const;

    //! Get bounds of the involute
    CELER_FUNCTION real_type tmin() const { return tmin_; }
    CELER_FUNCTION real_type tmax() const { return tmax_; }

    //! Get a view to the data for type-deleted storage
    CELER_FUNCTION StorageSpan data() const { return {&origin_[0], 6}; }

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
    Real2 origin_;

    // Involute parameters
    real_type r_b_;  // Radius, negative if "clockwise" (flipped)
    real_type a_;

    // Bounds
    real_type tmin_;
    real_type tmax_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from raw data.
 */
template<class R>
CELER_FUNCTION Involute::Involute(Span<R, StorageSpan::extent> data)
    : origin_{data[0], data[1]}
    , r_b_{data[2]}
    , a_{data[3]}
    , tmin_{data[4]}
    , tmax_{data[5]}
{
}

//---------------------------------------------------------------------------//
/*!
 * Orientation of the involute curve.
 */
CELER_FUNCTION auto Involute::sign() const -> Chirality
{
    return r_b_ > 0 ? Chirality::left : Chirality::right;
}

//---------------------------------------------------------------------------//
/*!
 * Determine the sense of the position relative to this surface.
 *
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
 * determined and used to calculate \em a via: \f[
 * a = angle - t_point
 * \f]
 *
 * To calculate the intercept of two circles it is assumed that the two circles
 * are \em x_prime axis. The coordinate along \em x_prime originating from the
 * center of one of the circles is given by: \f[ x_prime = (d^2 - r^2 +
 * R^2)/(2d) \f] Where \em d is the distance between the center of the two
 * circles, \em r is the radius of the displaced circle, and \em R is the
 * radius of the circle at the orgin. Then the point \em y_prime can be
 * obatined from: \f[ y_prime = \pm sqrt(R^2 - x_prime^2) \f] Then to convert (
 * \em x_prime , \em y_prime ) to ( \em x, \em y ) the following rotation is
 * applied:\f[ x = x_prime*cos(theta) - y_prime*sin(theta) y =
 * y_prime*cos(theta) + x_prime*sin(theta) \f] Where \em theta is the angle
 * between the \em x_prime axis and the \em x axis.
 */
CELER_FUNCTION SignedSense Involute::calc_sense(Real3 const& pos) const
{
    using constants::pi;

    Real2 xy;
    xy[0] = pos[0] - origin_[0];
    xy[1] = pos[1] - origin_[1];

    if (this->sign() == Chirality::right)
    {
        xy[0] = negate(xy[0]);
    }

    // Calculate distance to origin and obtain t value for distance.
    real_type const t_point_sq = dot_product(xy, xy) / ipow<2>(r_b_) - 1;

    // Check if point is in defined bounds.
    if (t_point_sq < ipow<2>(tmin_))
    {
        return SignedSense::outside;
    }
    if (t_point_sq > ipow<2>(tmax_))
    {
        return SignedSense::outside;
    }

    // Check if Point is on involute.
    detail::InvolutePoint calc_point{this->r_b(), a_};
    Real2 point = calc_point(clamp_to_nonneg(t_point_sq));

    if (xy == point)
    {
        return SignedSense::on;
    }

    // Check if point is in interval

    // Calculate tangent point
    // TODO: check that compiler avoids recomputing trig functions
    real_type x_prime = ipow<2>(r_b_) / norm(xy);
    real_type y_prime = std::sqrt(ipow<2>(r_b_) - ipow<2>(x_prime));

    // TODO: check that compiler avoids recomputing trig functions
    point[0] = (x_prime * xy[0] - y_prime * xy[1]) / norm(xy);
    point[1] = (y_prime * xy[0] + x_prime * xy[1]) / norm(xy);

    // Calculate angle of tangent
    real_type theta = std::acos(point[0] / norm(point));
    if (point[1] < 0)
    {
        theta = 2 * pi - theta;
    }
    // Count number of positive rotations around involute
    theta += max<real_type>(real_type{0},
                            std::floor((tmax_ + a_ - theta) / (2 * pi)))
             * 2 * pi;

    // Calculate the displacement angle of the point
    real_type a1 = theta - std::sqrt(clamp_to_nonneg(t_point_sq));

    // Check if point is inside bounds
    if (theta < tmax_ + a_ && a1 > a_)
    {
        return SignedSense::inside;
    }

    return SignedSense::outside;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate all possible straight-line intersections with this surface.
 */
CELER_FUNCTION auto Involute::calc_intersections(Real3 const& pos,
                                                 Real3 const& dir,
                                                 SurfaceState on_surface) const
    -> Intersections
{
    // Expand translated positions into 'xyz' coordinate system
    Real3 rel_pos{pos};
    rel_pos[0] -= origin_[0];
    rel_pos[1] -= origin_[1];

    detail::InvoluteSolver solve(this->r_b(), a_, this->sign(), tmin_, tmax_);

    return solve(rel_pos, dir, on_surface);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate outward normal at a position on the surface.
 *
 * This is done by taking the derivative of the involute equation : \f[
 * n = {\sin(t+a) - \cos(t+a), 0}
 * \f]
 */
CELER_FORCEINLINE_FUNCTION Real3 Involute::calc_normal(Real3 const& pos) const
{
    Real2 xy;
    xy[0] = pos[0] - origin_[0];
    xy[1] = pos[1] - origin_[1];

    // Calculate normal
    real_type const angle
        = std::sqrt(clamp_to_nonneg(dot_product(xy, xy) / ipow<2>(r_b_) - 1))
          + a_;
    Real3 normal_ = {std::sin(angle), -std::cos(angle), 0};

    if (this->sign() == Chirality::right)
    {
        normal_[0] = negate(normal_[0]);
    }

    return normal_;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
