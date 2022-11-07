//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/detail/QuadraticSolver.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/math/Algorithms.hh"
#include "orange/OrangeTypes.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Find positive, real, nonzero roots for quadratic functions.
 *
 * These are for quadratic functions \f[
   a x^2 + b^2 + c = 0
 * \f]
 * where a is nonzero (and not close to zero).
 *
 * This is used for all quadrics with potentially two roots (anything but
 * planes).
 *
 * Each item in the Intersections result will be a positive valid intersection
 * or the sentinel result \c no_intersection() .
 */
class QuadraticSolver
{
  public:
    //!@{
    //! Type aliases
    using Intersections = Array<real_type, 2>;
    //!@}

    //! Fuzziness for "along surface"
    static CELER_CONSTEXPR_FUNCTION real_type min_a() { return 1e-10; }

    // Solve when possibly along a surface (zeroish a)
    static inline CELER_FUNCTION Intersections solve_general(
        real_type a, real_type half_b, real_type c, SurfaceState on_surface);

  public:
    // Construct with nonzero a, and b/2
    inline CELER_FUNCTION QuadraticSolver(real_type a, real_type half_b);

    // Solve fully general case
    inline CELER_FUNCTION Intersections operator()(real_type c) const;

    // Solve degenerate case (known to be on surface)
    inline CELER_FUNCTION Intersections operator()() const;

  private:
    //// DATA ////
    real_type a_inv_;
    real_type hba_; // (b/2)/a
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Find all positive roots for general quadric surfaces.
 *
 * This is used for cones, simple quadrics, and general quadrics.
 */
CELER_FUNCTION auto QuadraticSolver::solve_general(real_type    a,
                                                   real_type    half_b,
                                                   real_type    c,
                                                   SurfaceState on_surface)
    -> Intersections
{
    if (std::fabs(a) >= min_a())
    {
        // Not along the surface
        QuadraticSolver solve(a, half_b);
        return on_surface == SurfaceState::on ? solve() : solve(c);
    }
    else
    {
        // Travelling parallel to the quadric's surface
        if (on_surface == SurfaceState::off)
        {
            QuadraticSolver solve(min_a(), half_b);
            return solve(c);
        }
        else
        {
            // On and along surface: no intersections
            return {no_intersection(), no_intersection()};
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Construct with b/2.
 */
CELER_FUNCTION QuadraticSolver::QuadraticSolver(real_type a, real_type half_b)
    : a_inv_(1 / a), hba_(half_b * a_inv_)
{
    CELER_EXPECT(std::fabs(a) >= min_a());
}

//---------------------------------------------------------------------------//
/*!
 * Find all positive roots of x^2 + (b/a)*x + (c/a) = 0.
 *
 * In this case, the quadratic formula can be written as: \f[
   x = -b/2 \pm \sqrt{(b/2)^2 - c}.
 * \f]
 *
 * Callers:
 * - General quadratic solve: not on nor along surface
 * - Sphere when not on surface
 * - Cylinder when not on surface
 */
CELER_FUNCTION auto QuadraticSolver::operator()(real_type c) const
    -> Intersections
{
    // Scale c by 1/a in accordance with scaling of b
    c *= a_inv_;
    real_type b2_4 = ipow<2>(hba_); // (b/2)^2

    Intersections result;

    if (b2_4 > c)
    {
        // Two real roots, r1 and r2
        real_type t2 = std::sqrt(b2_4 - c); // (b^2 - 4ac) / 4
        result[0]    = -hba_ - t2;
        result[1]    = -hba_ + t2;

        if (result[1] <= 0)
        {
            // Both are nonpositive
            result[0] = no_intersection();
            result[1] = no_intersection();
        }
        else if (result[0] <= 0)
        {
            // Only first is nonpositive
            result[0] = no_intersection();
        }
    }
    else if (b2_4 == c)
    {
        // One real root, r1
        result[0] = -hba_;
        result[1] = no_intersection();

        if (result[0] <= 0)
        {
            result[0] = no_intersection();
        }
    }
    else
    {
        // No real roots
        result = {no_intersection(), no_intersection()};
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Solve degenerate case (known to be "on" surface).
 *
 * Since x = 0 is a root, then c = 0 and x = -b is the other root. This will be
 * inaccurate if a particle is logically on a surface but not physically on it.
 */
CELER_FUNCTION auto QuadraticSolver::operator()() const -> Intersections
{
    Intersections result{-2 * hba_, no_intersection()};

    if (result[0] <= 0)
    {
        result[0] = no_intersection();
    }

    return result;
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
