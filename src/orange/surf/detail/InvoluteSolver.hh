//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/detail/InvoluteSolver.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>
#include <iostream>
#include <vector>

#include "corecel/Constants.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/math/Algorithms.hh"
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

  public:
    // Construct Involute from parameters
    inline CELER_FUNCTION InvoluteSolver(real_type r_b_,
                                         real_type a_,
                                         real_type sign_,
                                         real_type tmin_,
                                         real_type tmax_);

    // Solve fully general case
    inline CELER_FUNCTION Intersections operator()(Real3 const& pos,
                                                   Real3 const& dir) const;

    //// ACCESSORS ////

    //! Get involute parameters
    CELER_FUNCTION real_type r_b() const { return r_b_; }
    CELER_FUNCTION real_type a() const { return a_; }
    CELER_FUNCTION real_type sign() const { return mirror_; }

    //! Get bounds of the involute
    CELER_FUNCTION real_type tmin() const { return tmin_; }
    CELER_FUNCTION real_type tmax() const { return tmax_; }

  private:
    //// DATA ////
    // Involute parameters
    real_type r_b_;
    real_type a_;
    bool mirror_;

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
    real_type r_b, real_type a, real_type sign, real_type tmin, real_type tmax)
    : r_b_(r_b), a_(a), mirror_(sign < 0), tmin_(tmin), tmax_(tmax)
{
    CELER_EXPECT(r_b > 0);
    CELER_EXPECT(a > 0);
    CELER_EXPECT(std::fabs(tmax) < 2 * pi + std::fabs(tmin));

    if (mirror_)
    {
        a_ = -a + pi;
    }
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
 * ). The initial bounds can be determined by the set of: /f[ {0, beta - a,
 * beta - a - pi, beta - a + pi, beta - a - 2pi, beta - a + 2pi ...} /f] Where
 * \em beta is: \f[ beta = arctan(-v/u) \f]
 */
CELER_FUNCTION auto
InvoluteSolver::operator()(Real3 const& pos,
                           Real3 const& dir) const -> Intersections
{
    // Flatten pos and dir in xyz and uv respectively
    real_type x = pos[0];
    real_type const y = pos[1];
    real_type const z = pos[2];

    real_type u = dir[0];
    real_type const v = dir[1];
    real_type const w = dir[2];

    if (mirror_)
    {
        x = -x;
        u = -u;
    }

    // Lambda used for calculating the roots using Regular Falsi Iteration
    auto root = [](real_type t,
                   real_type x,
                   real_type y,
                   real_type u,
                   real_type v,
                   real_type r_b_,
                   real_type a_) {
        real_type a = u * std::sin(t + a_) - v * std::cos(t + a_);
        real_type b = t * (u * std::cos(t + a_) + v * std::sin(t + a_));
        real_type c = r_b_ * (a - b);
        return c + x * v - y * u;
    };
    /*
     * Results initalization
     */
    Intersections result;
    // Initial result vector.
    result = {no_intersection(), no_intersection(), no_intersection()};

    // Return result if particle is travelling along z-axis.
    if (u == 0 && v == 0)
    {
        return result;
    }

    // Initialize distance vector with root counter j
    Array<real_type, 3> dist;
    real_type j = 0;

    // Conversion constant for 2-D distance to 3-D distance

    real_type convert = std::sqrt(ipow<2>(v) + ipow<2>(u) + ipow<2>(w))
                        / std::sqrt(ipow<2>(v) + ipow<2>(u));

    /*
     * Define tolerances.
     * tol_point gives the tolerance level for a point and is set to 1e-7,
     * account for the floating point error when performing square roots.
     * tol_conv gives the tolerance for the Regular Falsi iteration,
     * ensuring that the root found produces a result that within 1e-8 of 0.
     */
    real_type const tol_point = 1e-7;
    real_type const tol_conv = 1e-8;

    // Check if particle is on a surface within tolerance

    real_type const rxy2 = ipow<2>(x) + ipow<2>(y);
    real_type const t_point = std::sqrt((rxy2 / (ipow<2>(r_b_))) - 1);
    real_type angle = t_point + a_;
    real_type x_inv = r_b_ * (std::cos(angle) + t_point * std::sin(angle));
    real_type y_inv = r_b_ * (std::sin(angle) - t_point * std::cos(angle));
    if (std::fabs(x - x_inv) < tol_point && std::fabs(y - y_inv) < tol_point)
    {
        dist[j] = 0;
        j++;
    }

    // Line angle parameter

    real_type beta;

    if (u != 0)
    {
        // Standard method
        beta = std::atan(-v / u);
    }  // Edge case
    else if (-v < 0)
    {
        beta = pi * -0.5;
    }
    else
    {
        beta = pi * 0.5;
    }

    // Setting first interval bounds, needs to be done to ensure roots are
    // found
    real_type t_lower = 0;
    real_type t_upper = beta - a_;

    // Round t_upper to the first positive multiple of pi
    t_upper += std::fmax(0, -std::floor(t_upper / pi)) * pi;

    // Parameters that will be used in loop
    int i = 1;
    real_type t_alpha;
    real_type t_beta;
    real_type t_gamma;
    real_type ft_gamma;
    real_type ft_alpha;
    real_type ft_beta;

    // Iterate on roots
    while (t_lower < tmax_)
    {
        // Set bounds on current iteration

        t_alpha = t_lower;
        t_beta = t_upper;

        // Find value in root function
        ft_alpha = root(t_lower, x, y, u, v, r_b_, a_);
        ft_beta = root(t_upper, x, y, u, v, r_b_, a_);

        // If bounds exceed tmax break
        if (t_lower > tmax_)
        {
            break;
        }

        // Only iterate when roots have different signs
        if ((0 < ft_alpha) - (ft_alpha < 0) != (0 < ft_beta) - (ft_beta < 0))
        {
            // Regula Falsi Iteration
            ft_gamma = 1;
            while (std::fabs(ft_gamma) >= tol_conv)
            {
                // Iterate on root
                t_gamma = (t_alpha * ft_beta - t_beta * ft_alpha)
                          / (ft_beta - ft_alpha);

                // Obtain root value of iterated root
                ft_gamma = root(t_gamma, x, y, u, v, r_b_, a_);

                // Update bounds with iterated root
                if ((0 < ft_beta) - (ft_beta < 0)
                    == (0 < ft_gamma) - (ft_gamma < 0))
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
            // Extract root and calculate point on involute
            angle = t_gamma + a_;
            x_inv = r_b_ * (std::cos(angle) + t_gamma * std::sin(angle));
            y_inv = r_b_ * (std::sin(angle) - t_gamma * std::cos(angle));

            // Check if point is interval
            if (std::fabs(t_gamma) >= std::fabs(tmin_) - tol_point
                && std::fabs(t_gamma) <= std::fabs(tmax_))
            {
                // Obatin direction to point on Involute
                real_type u2 = x_inv - x;
                real_type v2 = y_inv - y;

                // Dot with direction of particle
                real_type dot = u * u2 + v * v2;
                // Obtain distance to point
                real_type newdist = std::sqrt(ipow<2>(u2) + ipow<2>(v2));
                // Only record distance if dot product is positive
                if (dot >= 0 && newdist > tol_point)
                {
                    dist[j] = newdist;
                    j++;
                }
            }

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

        for (int k = 0; k < j; k++)
        {
            // Convert 2-D distances to 3-D distances
            result[k] = dist[k] * convert;
        }
    }
    return result;
}

}  // namespace detail
}  // namespace celeritas