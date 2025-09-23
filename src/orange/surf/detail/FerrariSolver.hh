//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/detail/FerrariSolver.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>
#include <iostream>

#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/math/Algorithms.hh"
#include "orange/OrangeTypes.hh"
#include "orange/surf/detail/QuadraticSolver.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Find positive, real, non-zero roots for quartic functions using the 
 * Ferrari-Cardano method.
 * 
 * The quartic equation \f[
 * a x^4 + b x^3 + c x^2 + d x + e = 0
 * \f]
 * has two solutions mathematically, but we only require solutions which are
 * both real and positive. This equation is also subject to multiple cases of
 * catastrophic precision-limitation-based error both fundamentally and as a
 * consequence of the particular algorithm chosen. This solver implements the
 * Ferrari-Cardano method, which is well-established and simple, but more
 * prone to numerical error than contemporary methods to be explored such as
 * Algorithm 1010.
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
        using Roots2 = Array<real_type, 2>;
        //!@}

        // General case solve
        static inline CELER_FUNCTION Intersections solve_general(
            real_type a, real_type b, real_type c, real_type d, real_type e, SurfaceState on_surface
        );

    public:
        // Construct w/ a, b, c, d
        inline CELER_FUNCTION FerrariSolver(real_type a, real_type b, real_type c, real_type d);

        // Solver fully general case 
        inline CELER_FUNCTION Intersections operator()(real_type e) const;

        // Solve degenerate / known surface case (e = 0)
        inline CELER_FUNCTION Intersections operator()() const;

        // Find dominant root of normalized cubic
        static inline CELER_FUNCTION real_type dominant_root_normalized_cubic(
            real_type b, real_type c, real_type d
        );

        // Find real quadratic roots TODO: is there a way to use existing one?
        static inline CELER_FUNCTION Roots2 real_roots_normalized_quadratic(
            real_type b, real_type c
        );

    private:
        //// DATA ////
        real_type a_inv_; // 1/a
        real_type ba_; // b/a
        real_type ca_; // c/a
        real_type da_; // d/a
        real_type ea_; // e/a

        //// UTIL ////
        // Place real root in an ascending list
        static inline CELER_FUNCTION void place_root_sorted(
            Intersections &roots, real_type new_root
        );

};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Find all positive roots for general quartic surfaces.
 * Uses the Ferrari-Cardano algorithm.
 * 
 * Currently, this is only used for toroids.
 */
CELER_FUNCTION auto FerrariSolver::solve_general(real_type a,
                                                 real_type b,
                                                 real_type c,
                                                 real_type d,
                                                 real_type e,
                                                 SurfaceState on_surface)
    -> Intersections
{
    FerrariSolver solve(a, b, c, d);
    if (on_surface == SurfaceState::on) {
        return solve();
    } else {
        return solve(e);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Default constructor with all five parameters a, b, c, d, and e.
 */
CELER_FUNCTION FerrariSolver::FerrariSolver(real_type a, real_type b, real_type c, real_type d)
    : a_inv_(1 / a), ba_(b * a_inv_), ca_(c * a_inv_), da_(d * a_inv_)
{
    
}

//---------------------------------------------------------------------------//
/*!
 * Find all positive roots of x^4 + (b/a)x^3 + (c/a)x^2 + (d/a)x + (e/a) = 0
 * 
 * Replaces negative or complex roots with no_intersection()
 */
CELER_FUNCTION auto FerrariSolver::operator()(real_type e) const
    -> Intersections
{
    real_type qb = 0.25*ba_;
    real_type qb2 = qb*qb;

    // Incomplete quartic
    real_type p = 3*qb2 - 0.5*ca_;
    real_type q = 4*qb*qb2 - ca_*qb + 0.5*da_;
    real_type r = 3*qb2*qb2 - ca_*qb2 + da_*qb - (e*a_inv_);

    // Final roots to return
    Intersections roots(no_intersection(), no_intersection(), no_intersection(), no_intersection());

    // Edge case: equation is biquadratic TODO: need biquadratic tolerance
    if (std::abs(q) <= 0) {
        // QuadraticSolver solve(1, -2*p);
        // auto ir = solve(-r);
        auto ir = real_roots_normalized_quadratic(-p, -r);
        // QuadraticSolver puts real & >0 roots last
        if (ir[1] != no_intersection() && ir[1] > 0) {
            real_type sqrt_ir1 = std::sqrt(ir[1]);
            real_type from_pos1 = sqrt_ir1 - qb;
            place_root_sorted(roots, from_pos1);
            if (from_pos1 > 0) {
                place_root_sorted(roots, -sqrt_ir1 - qb);
            }
        }
        if (ir[0] != no_intersection() && ir[0] > 0) {
            real_type sqrt_ir0 = std::sqrt(ir[0]);
            real_type from_pos0 = sqrt_ir0 - qb;
            place_root_sorted(roots, from_pos0);
            if (from_pos0 > 0) {
                place_root_sorted(roots, -sqrt_ir0 - qb);
            }
        }
        return roots;
    }

    // One real root of subsidiary cubic
    real_type z0 = FerrariSolver::dominant_root_normalized_cubic(p, r, p*r - 0.5*q*q);

    real_type s2 = 2*p + 2*z0;
    if (s2 >= 0) {
        real_type s = std::sqrt(s2);
        real_type t;
        if (s == 0) { // TODO: Is there a tolerance where it makes more sense to use s=0?
            t = z0*z0 + r;
        } else {
            t = -q / s;
        }
        auto roots01 = real_roots_normalized_quadratic(s*0.5, z0+t);
        auto roots23 = real_roots_normalized_quadratic(-s*0.5, z0-t);

        place_root_sorted(roots, roots01[0] - qb);
        place_root_sorted(roots, roots01[1] - qb);
        place_root_sorted(roots, roots23[0] - qb);
        place_root_sorted(roots, roots23[1] - qb);
    }
    return roots;
}

CELER_FUNCTION auto FerrariSolver::operator()() const -> Intersections
{
    // TODO: Either find an optimization to make here, or remove this entirely
    return operator()(0); 
}

//---------------------------------------------------------------------------//
/*!
 * Utility function which places the given real root into an intersection list 
 * in increasing order.
 * 
 */
CELER_FUNCTION void FerrariSolver::place_root_sorted(Intersections &roots, real_type new_root)
{
    if (new_root == no_intersection() || new_root <= 0) {
        return;
    }
    for (int i = 0; i < 4; i++) {
        if (roots[i] == no_intersection()) {
            roots[i] = new_root;
            break;
        } else if (new_root < roots[i]) {
            for (int j = 3; j > i; j--) {
                roots[j] = roots[j-1];
            }
            roots[i] = new_root;
            break;
        }
    }

} 

//---------------------------------------------------------------------------//
/*!
 * Utility function which solves for the dominant root of a cubic function.
 * Specifically, the cubic function \f[
 * a x^3 + b x^2 + c x + d
 * \f], where a is assumed to already be 1, and is not provided to the function.
 * 
 * \return The dominant real root of the given cubic equation.
 */
CELER_FUNCTION real_type FerrariSolver::dominant_root_normalized_cubic(real_type b, real_type c, real_type d)
{
    real_type third = 1.0 / 3.0;
    real_type third_b = b*third;
    real_type ninth_b2 = third_b*third_b;

    // Intermediate values
    real_type f = third*c - ninth_b2;
    real_type g = third_b * (2*ninth_b2 - c) + d;
    real_type h = 0.25*g*g + f*f*f;

    // degenerate case, NEEDS TOLERANCE FOR ZERO CHECK
    if (f == 0 && g == 0 && h == 0) {
        return -1.0 * std::cbrt(d);
    } else if (h <= 0) {
        real_type j = std::sqrt(-f);
        real_type k = std::acos(-0.5*g / (j*j*j));
        real_type m = std::cos(third*k);
        return 2*j*m - third_b;
    } else {
        real_type sqrt_h = std::sqrt(h);
        real_type s = std::cbrt(-0.5*g + sqrt_h);
        real_type u = std::cbrt(-0.5*g - sqrt_h);
        return s + u - third_b;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Utility function to return real roots of a quadratic function.
 * Specifically, \f[
 * a x^2 + (hb*2) x + c
 * \f], where a is assumed to already be 1 and not provided.
 * 
 * \return A pair of roots. If roots are imaginary, returns 2x no_intersection()
 */
CELER_FUNCTION auto FerrariSolver::real_roots_normalized_quadratic(real_type hb, real_type c) -> Roots2
{
    real_type qb2 = ipow<2>(hb);
    if (qb2 > c) {
        // Two real roots
        real_type ht = std::sqrt(qb2 - c);
        return Roots2(
            -hb - ht,
            -hb + ht
        );
    } else if (qb2 == c) {
        return Roots2(
            -hb,
            no_intersection()
        );
    } else {
        return Roots2(no_intersection(), no_intersection());
    }
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas