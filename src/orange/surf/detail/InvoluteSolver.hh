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
#include "orange/OrangeTypes.hh"

namespace celeritas
{
namespace detail
{
using constants::pi;
// Lambda used for calculating the roots using Regular Falsi Iteration
auto root = [](real_type t,
               real_type x,
               real_type y,
               real_type u,
               real_type v,
               real_type r_b,
               real_type a) {
    real_type alpha = u * std::sin(t + a) - v * std::cos(t + a);
    real_type beta = t * (u * std::cos(t + a) + v * std::sin(t + a));
    real_type gamma = r_b * (alpha - beta);
    return gamma + x * v - y * u;
};
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

    static inline CELER_FUNCTION real_type regular_falsi(real_type t_alpha,
                                                         real_type t_beta,
                                                         real_type ft_alpha,
                                                         real_type ft_beta,
                                                         real_type r_b,
                                                         real_type a,
                                                         real_type x,
                                                         real_type y,
                                                         real_type u,
                                                         real_type v,
                                                         real_type tol_conv);

    static inline CELER_FUNCTION real_type calc_dist(real_type x,
                                                     real_type y,
                                                     real_type u,
                                                     real_type v,
                                                     real_type t,
                                                     real_type r_b,
                                                     real_type a,
                                                     real_type tmin,
                                                     real_type tmax);

  public:
    // Construct Involute from parameters
    inline CELER_FUNCTION InvoluteSolver(
        real_type r_b, real_type a, Sign sign, real_type tmin, real_type tmax);

    // Solve fully general case
    inline CELER_FUNCTION Intersections operator()(
        Real3 const& pos, Real3 const& dir, SurfaceState on_surface) const;

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
    CELER_EXPECT(a > 0);
    CELER_EXPECT(tmax > 0);
    CELER_EXPECT(tmin > 0);
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
 * tc = [ta*f(tb) - tb*f(ta)] / [f(tb) - f(ta)]
 * \f]
 * where \em tc replaces the bound with the same sign (e.g. \em ta and \em tb
 * ). The initial bounds can be determined by the set of: \f[
 * {0, beta - a, beta - a - pi, beta - a + pi, beta - a - 2pi, beta - a + 2pi
 * ...} /f] Where \em beta is: \f[ beta = arctan(-v/u) \f]
 */
CELER_FUNCTION auto
InvoluteSolver::operator()(Real3 const& pos,
                           Real3 const& dir,
                           SurfaceState on_surface) const -> Intersections
{
    using Tolerance = celeritas::Tolerance<real_type>;

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
    Tolerance tol_point = Tolerance::from_relative(1e-7, r_b_);
    Tolerance tol_conv = Tolerance::from_relative(1e-8, r_b_);

    // Results initalization and root counter
    Intersections result;
    real_type j = 0;
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
    t_upper += std::fmax(0, -std::floor(t_upper / pi)) * pi;

    // Parameters that will be used in loop
    int i = 1;

    // Iterate on roots
    while (t_lower < tmax_)
    {
        // Set bounds on current iteration

        real_type t_alpha = t_lower;
        real_type t_beta = t_upper;

        // Find value in root function
        real_type ft_alpha = root(t_lower, x, y, u, v, r_b_, a_);
        real_type ft_beta = root(t_upper, x, y, u, v, r_b_, a_);

        // Only iterate when roots have different signs
        if ((0 < ft_alpha) - (ft_alpha < 0) != (0 < ft_beta) - (ft_beta < 0))
        {
            // Regular Falsi Iteration
            real_type t_gamma = regular_falsi(t_alpha,
                                              t_beta,
                                              ft_alpha,
                                              ft_beta,
                                              r_b_,
                                              a_,
                                              x,
                                              y,
                                              u,
                                              v,
                                              tol_conv.abs);

            // Convert root to distance and store if positive and in interval
            real_type dist
                = calc_dist(x, y, u, v, t_gamma, r_b_, a_, tmin_, tmax_);
            if (dist > tol_point.abs)
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
 * Perform single Regular Falsi Iteration
 */
CELER_FUNCTION real_type InvoluteSolver::regular_falsi(real_type t_alpha,
                                                       real_type t_beta,
                                                       real_type ft_alpha,
                                                       real_type ft_beta,
                                                       real_type r_b,
                                                       real_type a,
                                                       real_type x,
                                                       real_type y,
                                                       real_type u,
                                                       real_type v,
                                                       real_type tol_conv)
{
    real_type ft_gamma = 1;
    real_type t_gamma;

    while (std::fabs(ft_gamma) >= tol_conv)
    {
        // Iterate on root
        t_gamma = (t_alpha * ft_beta - t_beta * ft_alpha)
                  / (ft_beta - ft_alpha);

        // Obtain root value of iterated root
        ft_gamma = root(t_gamma, x, y, u, v, r_b, a);

        // Update bounds with iterated root
        if ((0 < ft_beta) - (ft_beta < 0) == (0 < ft_gamma) - (ft_gamma < 0))
        {
            t_beta = t_gamma;
            ft_beta = ft_gamma;
        }
        else
        {
            t_alpha = t_gamma;
            ft_alpha = ft_gamma;
        }
    }
    return t_gamma;
}

/*!
 * Convert root to distance by calculating the point on the involute given by
 * the root and then taking the distance to that point from the particle.
 */
CELER_FUNCTION real_type InvoluteSolver::calc_dist(real_type x,
                                                   real_type y,
                                                   real_type u,
                                                   real_type v,
                                                   real_type t,
                                                   real_type r_b,
                                                   real_type a,
                                                   real_type tmin,
                                                   real_type tmax)
{
    real_type angle = t + a;
    real_type x_inv = r_b * (std::cos(angle) + t * std::sin(angle));
    real_type y_inv = r_b * (std::sin(angle) - t * std::cos(angle));
    real_type dist = 0;

    // Check if point is interval
    if (t >= tmin && t <= tmax)
    {
        // Obatin direction to point on Involute
        real_type u_point = x_inv - x;
        real_type v_point = y_inv - y;

        // Dot with direction of particle
        real_type dot = u * u_point + v * v_point;
        real_type dot_sign = (0 < dot) - (dot < 0);
        // Obtain distance to point
        dist = std::sqrt(ipow<2>(u_point) + ipow<2>(v_point)) * dot_sign;
    }
    return dist;
}

}  // namespace detail
}  // namespace celeritas