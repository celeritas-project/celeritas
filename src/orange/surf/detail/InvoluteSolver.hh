//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/detail/InvoluteSolver.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Constants.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/BisectionRootFinder.hh"
#include "corecel/math/IllinoisRootFinder.hh"
#include "corecel/math/RegulaFalsiRootFinder.hh"
#include "orange/OrangeTypes.hh"

namespace celeritas
{
namespace detail
{
using constants::pi;
//---------------------------------------------------------------------------//
/*!
 * Find positive, real, nonzero roots for involute intersection function.
 *
 * The involute intersection equation \f[
   r_b * [v{cos(t+a)+tsin(t+a)} + u{sin(t+a)-tcos(t+a)}] + xv - yu = 0
 * \f]
 * has n solutions mathematically, but we only want solutions where t results
 * in a real, positive, and in the defined bounds.  Furthermore the equation is
 * subject to catastrophic roundoff due to floating point precision (see
 * \c Tolerance::sqrt_quadratic and the derivation in \c CylAligned ).
 *
 * \return An Intersections array where each item is a positive valid
 * intersection or the sentinel result \c no_intersection() .
 */
class InvoluteSolver
{
  public:
    //!@{
    //! \name Type aliases
    using Intersections = Array<real_type, 3>;
    using SurfaceSate = celeritas::SurfaceState;

    //! Enum defining chirality of involute
    enum Sign : bool
    {
        counterclockwise = 0,
        clockwise,  //!< Apply symmetry when solving
    };

    static inline CELER_FUNCTION real_type line_angle_param(real_type u,
                                                            real_type v);

  public:
    // Construct Involute from parameters
    inline CELER_FUNCTION InvoluteSolver(
        real_type r_b, real_type a, Sign sign, real_type tmin, real_type tmax);

    // Solve fully general case
    inline CELER_FUNCTION Intersections operator()(
        Real3 const& pos, Real3 const& dir, SurfaceState on_surface) const;

    //// CALCULATION ////
    inline CELER_FUNCTION real_type calc_dist(
        real_type x, real_type y, real_type u, real_type v, real_type t) const;

    static CELER_CONSTEXPR_FUNCTION real_type tol()
    {
        if constexpr (std::is_same_v<real_type, double>)
            return 1e-8;
        else if constexpr (std::is_same_v<real_type, float>)
            return 1e-5;
    }

    //// ACCESSORS ////

    //! Get involute parameters
    CELER_FUNCTION real_type r_b() const { return r_b_; }
    CELER_FUNCTION real_type a() const { return a_; }
    CELER_FUNCTION Sign sign() const { return sign_; }

    //! Get bounds of the involute
    CELER_FUNCTION real_type tmin() const { return tmin_; }
    CELER_FUNCTION real_type tmax() const { return tmax_; }

  private:
    //// DATA ////
    // Involute parameters
    real_type r_b_;
    real_type a_;
    Sign sign_;

    // Bounds
    real_type tmin_;
    real_type tmax_;
};
//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
/*!
 * Construct from involute parameters.
 */
CELER_FUNCTION InvoluteSolver::InvoluteSolver(
    real_type r_b, real_type a, Sign sign, real_type tmin, real_type tmax)
    : r_b_(r_b), a_(a), sign_(sign), tmin_(tmin), tmax_(tmax)
{
    CELER_EXPECT(r_b > 0);
    CELER_EXPECT(a >= 0);
    CELER_EXPECT(tmax > 0);
    CELER_EXPECT(tmin >= 0);
    CELER_EXPECT(tmax < 2 * pi + tmin);
}

//---------------------------------------------------------------------------//

/*!
 * Find all roots for involute surfaces that are within the bounds and result
 * in positive distances. Performed by doing a Regular Falsi Iteration on the
 * root function, \f[
 * f(t) = r_b * [v{cos(a+t) + tsin(a+t)} + u{sin(a+t) - tcos(a+t)}] + xv - yu
 * \f]
 * where the Regular Falsi Iteration is given by: \f[
 * t_c = [t_a*f(t_b) - t_b*f(t_a)] / [f(t_b) - f(t_a)]
 * \f]
 * where \em t_c replaces the bound with the same sign (e.g. \em t_a and \em
 * t_b
 * ). The initial bounds can be determined by the set of: \f[
 * {0, beta - a, beta - a - pi, beta - a + pi, beta - a - 2pi, beta - a + 2pi
 * ...} /f] Where \em beta is: \f[ beta = arctan(-v/u) \f]
 */
CELER_FUNCTION auto
InvoluteSolver::operator()(Real3 const& pos,
                           Real3 const& dir,
                           SurfaceState on_surface) const -> Intersections
{
    // Flatten pos and dir in xyz and uv respectively
    real_type x = pos[0];
    real_type const y = pos[1];

    real_type u = dir[0];
    real_type v = dir[1];

    // Mirror systemm for counterclockwise involutes
    if (sign_)
    {
        x = -x;
        u = -u;
    }

    /*
     * Define tolerances.
     * tol_point gives the tolerance level for a point,
     * account for the floating point error when performing square roots.
     * tol_conv gives the tolerance for the Regular Falsi iteration.
     */
    real_type tol_conv = r_b_ * tol();
    real_type tol_point = r_b_ * tol() * 100;

    // Results initalization and root counter
    Intersections result;
    int j = 0;
    // Initial result vector.
    result = {no_intersection(), no_intersection(), no_intersection()};

    // Return result if particle is travelling along z-axis.
    if (u == 0 && v == 0)
    {
        return result;
    }

    // Conversion constant for 2-D distance to 3-D distance

    real_type convert = 1 / std::sqrt(ipow<2>(v) + ipow<2>(u));
    u *= convert;
    v *= convert;

    // Remove 0 dist if particle is on surface
    if (on_surface == SurfaceState::on)
    {
        result[j] = 0;
        j++;
    }

    // Line angle parameter

    real_type beta = line_angle_param(u, v);

    // Setting first interval bounds, needs to be done to ensure roots are
    // found
    real_type t_lower = 0;
    real_type t_upper = beta - a_;

    // Round t_upper to the first positive multiple of pi
    real_type floor_upper = -std::floor(t_upper / pi);
    t_upper += max<real_type>(0, floor_upper) * pi;

    // Parameters that will be used in loop
    int i = 1;

    // Lambda used for calculating the roots using Regular Falsi Iteration
    auto calc_t_intersect = [&](real_type t) {
        real_type alpha = u * std::sin(t + a_) - v * std::cos(t + a_);
        real_type beta = t * (u * std::cos(t + a_) + v * std::sin(t + a_));
        real_type gamma = r_b_ * (alpha - beta);
        return gamma + x * v - y * u;
    };
    IllinoisRootFinder find_root_between{calc_t_intersect, tol_conv};

    // Iterate on roots
    while (t_lower < tmax_)
    {
        // Find value in root function
        real_type ft_lower = calc_t_intersect(t_lower);
        real_type ft_upper = calc_t_intersect(t_upper);

        // Only iterate when roots have different signs
        if (signum<real_type>(ft_lower) != signum<real_type>(ft_upper))
        {
            // Regular Falsi Iteration: Sometimes will slowly converge
            real_type t_gamma = find_root_between(t_lower, t_upper);

            // Convert root to distance and store if positive and in interval
            real_type dist = calc_dist(x, y, u, v, t_gamma);
            if (dist > tol_point
                || (!(on_surface == SurfaceState::on) && (dist > 0)))
            {
                result[j] = convert * dist;
                j++;
            }

            // Obtain next bounds
            t_lower = t_upper;
            t_upper += pi;
        }
        else
        {
            // Incremet interval slowly until root is in interval
            // i used to slow down approach to next root
            t_lower = t_upper;
            t_upper += pi / i;
            i++;
        }
    }
    return result;
}

/*!
 * Calculate the line-angle parameter \em beta used to find bounds of roots.
 * \f[ beta = arctan(-v/u) \f]
 */
CELER_FUNCTION real_type InvoluteSolver::line_angle_param(real_type u,
                                                          real_type v)
{
    if (u != 0)
    {
        // Standard method
        return std::atan(-v / u);
    }
    else if (-v < 0)
    {
        // Edge case
        return pi * -0.5;
    }
    else
    {
        return pi * 0.5;
    }
}

/*!
 * Convert root to distance by calculating the point on the involute given by
 * the root and then taking the distance to that point from the particle.
 */
CELER_FUNCTION real_type InvoluteSolver::calc_dist(
    real_type x, real_type y, real_type u, real_type v, real_type t) const
{
    real_type theta = t + a_;
    real_type x_inv = r_b_ * (std::cos(theta) + t * std::sin(theta));
    real_type y_inv = r_b_ * (std::sin(theta) - t * std::cos(theta));
    real_type dist = 0;

    // Check if point is interval
    if (t >= tmin_ && t <= tmax_)
    {
        // Obatin direction to point on Involute
        real_type u_point = x_inv - x;
        real_type v_point = y_inv - y;

        // Dot with direction of particle
        real_type dot = u * u_point + v * v_point;
        real_type dot_sign = signum<real_type>(dot);
        // Obtain distance to point
        dist = std::sqrt(ipow<2>(u_point) + ipow<2>(v_point)) * dot_sign;
    }
    return dist;
}

}  // namespace detail
}  // namespace celeritas