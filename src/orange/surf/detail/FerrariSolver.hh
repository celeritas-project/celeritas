//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/detail/FerrariSolver.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>
#include <iostream>

#include "corecel/Constants.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/PolyEvaluator.hh"
#include "corecel/math/SoftEqual.hh"
#include "orange/OrangeTypes.hh"
#include "orange/surf/detail/QuadraticSolver.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Find positive, real, non-zero roots for quartic functions using the
 * Ferrari-Cardano method\citet{polyanin-ferrari-2007,
 * https://doi.org/10.1201/9781420010510}.
 *
 * The quartic equation
 * \f[
   a x^4 + b x^3 + c x^2 + d x + e = 0
 * \f]
 * has four solutions mathematically, but we only require solutions which are
 * both real and positive. This equation is also subject to multiple cases of
 * catastrophic precision-limitation-based error both fundamentally and as a
 * consequence of the particular algorithm chosen. This solver implements the
 * Ferrari-Cardano method, which is well-established and simple, but more
 * prone to numerical error than contemporary methods to be explored such as
 * Algorithm 1010\citet{orellana-alg1010-2020,
 * https://doi.org/10.1145/3386241}.
 *
 * \return An Intersections array where each item is either a positive valid
 * intersection or the sentinel result \c no_intersection().
 */
class FerrariSolver
{
  public:
    //!@{
    //! \name Type aliases
    using Intersections = Array<real_type, 4>;
    using Real2 = Array<real_type, 2>;
    using Real3 = Array<real_type, 3>;
    using Real4 = Array<real_type, 4>;
    using Real5 = Array<real_type, 5>;
    //!@}

  public:
    // Construct w/ given tolerance. Uses quadratic tolerance by default.
    inline CELER_FUNCTION
    FerrariSolver(real_type tolerance = Tolerance<real_type>::sqrt_quadratic());

    // Solver for fully general case
    inline CELER_FUNCTION Intersections operator()(Real5 const& abcde,
                                                   SurfaceState on_surface
                                                   = SurfaceState::off) const;

    // Solver for surface case
    inline CELER_FUNCTION Intersections operator()(Real4 const& abcd) const;

  private:
    //// DATA ////
    // Soft zero for biquadratic and degenerate cubic detection
    SoftZero<real_type> const soft_zero_;

    //// UTIL ////
    // Try to place real at given index in list, return next free index
    inline CELER_FUNCTION int
    place_root(Intersections& roots, real_type new_root, int free_index) const;

    // Find roots of special reduced quartic which is biquadratic
    inline CELER_FUNCTION Intersections
    calc_biquadratic_roots(real_type qb, real_type p, real_type r) const;

    // Find all roots of normalized cubic (unsorted, dominant first)
    inline CELER_FUNCTION Real3 real_roots_normalized_cubic(real_type b,
                                                            real_type c,
                                                            real_type d) const;

    // Find real quadratic roots
    inline CELER_FUNCTION Real2
    real_roots_normalized_quadratic(real_type b, real_type c) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
/*!
 * Construct a solver instance with a specified tolerance for degenerate cases,
 * such as the particle starting on the surface.
 */
CELER_FUNCTION
FerrariSolver::FerrariSolver(real_type tolerance)
{
    SoftZero soft_zero_{tolerance};
}

//---------------------------------------------------------------------------//
/*!
 * Find all positive roots of the polynomial with given a, b, c, d, e.
 *
 * The polynomial takes the form:
 * \f[
   ax^4 + bx^3 + cx^2 + dx + e = 0.
 *\f]
 * Where the given array abcde corresponds to {a, b, c, d, e}.
 * Replaces negative or complex roots with sentry value no_intersection().
 *
 * Allows user to specify if the operation takes place on the surface, in which
 * case it will solve as a cubic to avoid the root at 0.
 */
CELER_FUNCTION auto
FerrariSolver::operator()(Real5 const& abcde, SurfaceState on_surface) const
    -> Intersections
{
    // Normalize coefficients
    auto [a, b, c, d, e] = abcde;
    real_type ba = b / a, ca = c / a, da = d / a, ea = e / a;

    // If known to be on surface, divide polynomial by x and solve cubic
    if (on_surface == SurfaceState::on)
    {
        Real3 cubic_roots = real_roots_normalized_cubic(ba, ca, da);
        auto [z0, z1, z2] = cubic_roots;

        Intersections roots(no_intersection(),
                            no_intersection(),
                            no_intersection(),
                            no_intersection());

        int idx = 0;
        idx = place_root(roots, z0, idx);
        idx = place_root(roots, z1, idx);
        idx = place_root(roots, z2, idx);
        sort(&roots[0], &roots[idx]);
        return roots;
    }

    constexpr real_type half{0.5};
    real_type qb = real_type{0.25} * ba;

    // Incomplete quartic
    real_type p = PolyEvaluator{-half * ca, 0, 3}(qb);
    real_type q = PolyEvaluator{half * da, -ca, 0, 4}(qb);
    real_type r = PolyEvaluator{-ea, da, -ca, 0, 3}(qb);

    // Edge case: equation is biquadratic
    if (soft_zero_(q))
    {
        return calc_biquadratic_roots(qb, p, r);
    }

    // One real root of subsidiary cubic
    Real3 z = FerrariSolver::real_roots_normalized_cubic(
        p, r, p * r - half * ipow<2>(q));
    real_type z0 = z[0];

    real_type s2 = 2 * p + 2 * z0;
    if (s2 >= 0)
    {
        real_type s = std::sqrt(s2);
        real_type t;
        if (soft_zero_(s))
        {
            t = ipow<2>(z0) + r;
        }
        else
        {
            t = -q / s;
        }
        auto const [r0, r1] = real_roots_normalized_quadratic(s * half, z0 + t);
        auto const [r2, r3]
            = real_roots_normalized_quadratic(-s * half, z0 - t);

        Intersections roots(no_intersection(),
                            no_intersection(),
                            no_intersection(),
                            no_intersection());
        int idx = 0;
        idx = place_root(roots, r0 - qb, idx);
        idx = place_root(roots, r1 - qb, idx);
        idx = place_root(roots, r2 - qb, idx);
        idx = place_root(roots, r3 - qb, idx);

        sort(&roots[0], &roots[idx]);

        return roots;
    }
    else
    {
        return Intersections(no_intersection(),
                             no_intersection(),
                             no_intersection(),
                             no_intersection());
    }
}

//---------------------------------------------------------------------------//
/*!
 * Overload to solve a quartic polynomial where coefficient e is known to be 0.
 *
 * The polynomial takes the form:
 * \f[
   ax^4 + bx^3 + cx^2 + dx = 0.
 *\f]
 * Where the given array abcd corresponds to {a, b, c, d}.
 * Replaces negative or complex roots with sentry value no_intersection().
 *
 * Solves as cubic equation, and does not return the known root of 0.
 */
CELER_FUNCTION auto FerrariSolver::operator()(Real4 const& abcd) const
    -> Intersections
{
    auto [a, b, c, d] = abcd;
    return operator()({a, b, c, d, 0}, SurfaceState::on);
}

//---------------------------------------------------------------------------//
/*!
 * Attempt to put a value into the given list at given index, returning where
 * to place the next item.
 *
 * If the given value is no_intersection() or is not positive, does not place
 * the root, and returns the same index for the next one.
 */
CELER_FUNCTION int FerrariSolver::place_root(Intersections& roots,
                                             real_type new_root,
                                             int free_index) const
{
    if (!(new_root == no_intersection() || new_root <= 0))
    {
        roots[free_index] = new_root;
        free_index += 1;
    }
    return free_index;
}

//---------------------------------------------------------------------------//
/*!
 * Solve special case of Ferrari where reduced quartic is also biquadratic.
 *
 * In this special case, the normal solution won't work, and must instead be
 * solved as a quadratic equation: The square roots of each quadratic solution
 * then go on to form potential quartic solutions, for up to four roots.
 */
CELER_FUNCTION auto FerrariSolver::calc_biquadratic_roots(real_type qb,
                                                          real_type p,
                                                          real_type r) const
    -> Intersections
{
    auto ir = real_roots_normalized_quadratic(-p, -r);
    Intersections roots(no_intersection(),
                        no_intersection(),
                        no_intersection(),
                        no_intersection());
    int idx = 0;
    if (ir[1] != no_intersection() && ir[1] > 0)
    {
        real_type sqrt_ir1 = std::sqrt(ir[1]);
        real_type from_pos1 = sqrt_ir1 - qb;
        idx = place_root(roots, from_pos1, idx);
        if (from_pos1 > 0)
        {
            idx = place_root(roots, -sqrt_ir1 - qb, idx);
        }
    }
    if (ir[0] != no_intersection() && ir[0] > 0)
    {
        real_type sqrt_ir0 = std::sqrt(ir[0]);
        real_type from_pos0 = sqrt_ir0 - qb;
        idx = place_root(roots, from_pos0, idx);
        if (from_pos0 > 0)
        {
            idx = place_root(roots, -sqrt_ir0 - qb, idx);
        }
    }
    sort(&roots[0], &roots[idx]);
    return roots;
}

//---------------------------------------------------------------------------//
/*!
 * Solve for the real roots of a cubic function.
 *
 * Specifically, the cubic function
 * \f[
   a x^3 + b x^2 + c x + d
 * \f]
 * where a is assumed to already be 1, and is not provided to the
 * function.
 * Uses the Numerical Recipes cubic algorithm, a combination of Cardano and
 * trigonometry.
 *
 * \return The real roots of the given cubic equation, with the dominant at
 * index 0.
 */
CELER_FUNCTION auto
FerrariSolver::real_roots_normalized_cubic(real_type b,
                                           real_type c,
                                           real_type d) const -> Real3
{
    constexpr real_type half = real_type{0.5};
    constexpr real_type third = real_type{1} / real_type{3};
    real_type third_b = b * third;

    // Intermediate values
    real_type q = ipow<2>(third_b) - third * c;
    real_type r = half * PolyEvaluator{d, -c, 0, 2}(third_b);

    real_type q3 = ipow<3>(q);
    real_type r2 = ipow<2>(r);

    real_type discrim = r2 - q3;

    if (soft_zero_(q) && soft_zero_(r) && soft_zero_(discrim))
    {
        return Real3(-std::cbrt(d), no_intersection(), no_intersection());
    }
    else if (discrim <= 0)  // All roots real, calculate with trigomonetry
    {
        real_type theta = std::acos(r / std::sqrt(q3));
        real_type n2_root_q = real_type{-2} * std::sqrt(q);
        real_type twth_pi = constants::pi * real_type{2} * third;
        real_type third_theta = theta * third;

        real_type z0 = n2_root_q * std::cos(third_theta) - third_b;
        real_type z1 = n2_root_q * std::cos(third_theta + twth_pi) - third_b;
        real_type z2 = n2_root_q * std::cos(third_theta - twth_pi) - third_b;

        if (real_type{2} * theta < constants::pi)
        {
            return Real3(z0, z1, z2);
        }
        else
        {
            return Real3(z1, z0, z2);
        }
    }
    else  // One real and two complex roots, solve for real root with Cardano
    {
        real_type nr_a = -signum(r)
                         * std::cbrt(std::abs(r) + std::sqrt(discrim));
        real_type nr_b = nr_a == 0 ? 0 : q / nr_a;
        real_type z0 = nr_a + nr_b - third_b;
        Real3 z(z0, no_intersection(), no_intersection());
        if (soft_zero_(discrim))
        {
            // If near double root, accept within quadratic tolerance
            // TODO: is this actually necessary? How bad is it to not graze the
            // side?
            z[1] = -half * (nr_a + nr_b) - third_b;
        }
        return z;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Solve for the real roots of a quadratic function.
 *
 * Specifically, the quadratic function
 * \f[
   a x^2 + (hb*2) x + c
 * \f]
 * where a is assumed to already be 1 and not provided.
 *
 * \return A pair of roots. If roots are imaginary, returns 2x
 * no_intersection().
 */
CELER_FUNCTION auto
FerrariSolver::real_roots_normalized_quadratic(real_type hb, real_type c) const
    -> Real2
{
    real_type qb2 = ipow<2>(hb);
    if (soft_zero_(qb2 - c))
    {
        // One critical root
        return Real2(-hb, no_intersection());
    }
    else if (qb2 > c)
    {
        // Two real roots
        real_type ht = std::sqrt(qb2 - c);
        return Real2(-hb - ht, -hb + ht);
    }
    else
    {
        return Real2(no_intersection(), no_intersection());
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
