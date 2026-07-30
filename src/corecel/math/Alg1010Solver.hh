//------------------------------- -*- C++ -*- -------------------------------//
// SPDX-FileCopyrightText: 2020 Alberto G. Orellana and Cristiano De Michele
// SPDX-FileCopyrightText: 2026 Celeritas contributors
// SPDX-License-Identifier: Apache-2.0
//---------------------------------------------------------------------------//
/*!
 * \file corecel/math/Alg1010Solver.hh
 * \brief Quartic solver derived from OpenMC modification of Algorithm 1010
 *
 * OpenMC C++ modification source:
 * https://github.com/openmc-dev/openmc/blob/a8152672be6cf5da38713281f0eb2e86d63663db/src/external/quartic_solver.cpp
 *
 * Original C Source, from publication of Algorithm 1010:
 * https://calgo.acm.org/ (Entry 1010)
 *
 * Original License (BSD) reproduced at end of file.
 */
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>
#include <complex>
#include <cstdlib>
#include <limits>

#include "corecel/Constants.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Range.hh"
#include "corecel/math/Complex.hh"
#include "corecel/math/NumericLimits.hh"
#include "corecel/math/PolyEvaluator.hh"
#include "corecel/math/SoftEqual.hh"

namespace celeritas
{
/*!
 * Find positive, real roots for quartic functions using Algorithm 1010.
 *
 * The quartic equation
 * \f[
   a x^4 + b x^3 + c x^2 + d x + e = 0
 * \f]
 * has four solutions mathematically, but we only require solutions which are
 * both real and positive.
 *
 * The input argument to an instance of this class is an array \c abcde that
 * corresponds to {a, b, c, d, e}. (The overload using a four-element array
 * \c abcd , for the degenerate case where \f$ e = 0 \f$, is presently only
 * for consistency, and not a direct call to a cubic solver.)
 *
 * The result is an array of 4 real numbers, where each is either a positive
 * valid intersection or the sentinel result \c infinity.
 *
 * This quartic solver uses Algorithm 1010\citep{orellana-alg1010-2020,
 * https://doi.org/10.1145/3386241}, a robust algorithm optimized for both
 * CPU performance and accuracy of solutions.
 *
 * Made referencing an existing implementation in OpenMC.
 *
 */
class Alg1010Solver
{
  public:
    //!@{
    //! \name Type aliases
    using Real4 = Array<real_type, 4>;
    using Real5 = Array<real_type, 5>;
    using result_type = Real4;
    using cmplx_type = complex;
    using Comp2 = Array<cmplx_type, 2>;
    using Comp3 = Array<cmplx_type, 3>;
    using Comp4 = Array<cmplx_type, 4>;
    //!@}

  public:
    // pow(DBL_MAX,1.0/3.0)/1.618034;
    static constexpr real_type CUBIC_RESCAL_FACT = 3.488062113727083e+102;
    // pow(DBL_MAX,1.0/4.0)/1.618034;
    static constexpr real_type QUART_RESCAL_FACT = 7.156344627944542e+76;
    static constexpr real_type MACHEPS
        = celeritas::numeric_limits<real_type>::epsilon();

    // Construct with given tolerance
    inline CELER_FUNCTION Alg1010Solver(real_type tolerance);

    //! Construct with default tolerance equal to ORANGE tolerance.
    inline CELER_FUNCTION Alg1010Solver() : Alg1010Solver{default_tol_} {}

    // Solver for fully general case
    inline CELER_FUNCTION result_type operator()(Real5 const& abcde) const;

    // Solver for surface case
    inline CELER_FUNCTION result_type operator()(Real4 const& abcd) const;

  private:
    //// TYPES ////

    //// STATIC DATA ////

    //! Default tolerance for quadric solve, taken from Orange `Tolerance`.
    static constexpr real_type default_tol_
        = (std::is_same_v<real_type, double> ? 1e-5 : 5e-2f);

    //! No positive real solution (aka "no intersection")
    static constexpr real_type no_solution_
        = NumericLimits<real_type>::infinity();

    //// DATA ////

    // Soft zero for biquadratic and degenerate cubic detection
    SoftZero<real_type> const soft_zero_;

    //// HELPER FUNCTIONS ////

    inline CELER_FUNCTION real_type solve_cubic_analytic_depressed_handle_inf(
        real_type b, real_type c) const;

    inline CELER_FUNCTION real_type solve_cubic_analytic_depressed(
        real_type b, real_type c) const;

    inline CELER_FUNCTION real_type calc_phi0(Real4 const& abcd,
                                              bool scaled) const;

    inline CELER_FUNCTION real_type calc_err_ldlt(real_type b,
                                                  real_type c,
                                                  real_type d,
                                                  real_type d2,
                                                  real_type l1,
                                                  real_type l2,
                                                  real_type l3) const;

    inline CELER_FUNCTION real_type calc_err_abcd(real_type a,
                                                  real_type b,
                                                  real_type c,
                                                  real_type d,
                                                  real_type aq,
                                                  real_type bq,
                                                  real_type cq,
                                                  real_type dq) const;

    inline CELER_FUNCTION real_type calc_err_abcd_cmplx(real_type a,
                                                        real_type b,
                                                        real_type c,
                                                        real_type d,
                                                        cmplx_type aq,
                                                        cmplx_type bq,
                                                        cmplx_type cq,
                                                        cmplx_type dq) const;

    inline CELER_FUNCTION real_type calc_err_abc(real_type a,
                                                 real_type b,
                                                 real_type c,
                                                 real_type aq,
                                                 real_type bq,
                                                 real_type cq,
                                                 real_type dq) const;

    inline CELER_FUNCTION void NRabcd(real_type a,
                                      real_type b,
                                      real_type c,
                                      real_type d,
                                      real_type* AQ,
                                      real_type* BQ,
                                      real_type* CQ,
                                      real_type* DQ) const;

    inline CELER_FUNCTION void
    solve_quadratic(real_type a, real_type b, Comp2& roots) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//

CELER_FUNCTION Alg1010Solver::Alg1010Solver(real_type tolerance)
    : soft_zero_{tolerance}
{
}

CELER_FUNCTION real_type Alg1010Solver::solve_cubic_analytic_depressed_handle_inf(
    real_type b, real_type c) const
{
    /* find analytically the dominant root of a depressed cubic x^3+b*x+c
     * where coefficients b and c are large (see sec. 2.2 in the manuscript) */
    real_type Q = -b / 3.0;
    real_type R = 0.5 * c;
    if (R == 0)
    {
        return (b <= 0) ? std::sqrt(-b) : 0;
    }

    real_type KK;
    if (std::fabs(Q) < std::fabs(R))
    {
        real_type QR = Q / R;
        real_type QRSQ = ipow<2>(QR);
        KK = 1.0 - Q * QRSQ;
    }
    else
    {
        real_type RQ = R / Q;
        KK = std::copysign(1.0, Q) * (ipow<2>(RQ) / Q - 1.0);
    }

    if (KK < 0.0)
    {
        real_type sqrtQ = std::sqrt(Q);
        real_type theta = std::acos((R / std::fabs(Q)) / sqrtQ);
        if (2.0 * theta < M_PI)
            return -2.0 * sqrtQ * std::cos(theta / 3.0);
        else
            return -2.0 * sqrtQ * std::cos((theta + 2.0 * M_PI) / 3.0);
    }
    else
    {
        real_type A;
        if (std::fabs(Q) < std::fabs(R))
            A = -std::copysign(1.0, R)
                * cbrt(std::fabs(R) * (1.0 + std::sqrt(KK)));
        else
        {
            A = -std::copysign(1.0, R)
                * cbrt(
                    std::fabs(R)
                    + std::sqrt(std::fabs(Q)) * std::fabs(Q) * std::sqrt(KK));
        }
        real_type B = (A == 0.0) ? 0.0 : Q / A;
        return A + B;
    }
}

CELER_FUNCTION real_type Alg1010Solver::solve_cubic_analytic_depressed(
    real_type b, real_type c) const
{
    /* find analytically the dominant root of a depressed cubic x^3+b*x+c
     * (see sec. 2.2 in the manuscript) */
    real_type Q = -b / 3.0;
    real_type R = 0.5 * c;
    if (std::fabs(Q) > 1e102 || std::fabs(R) > 1e154)
    {
        return Alg1010Solver::solve_cubic_analytic_depressed_handle_inf(b, c);
    }
    real_type Q3 = ipow<3>(Q);
    real_type R2 = ipow<2>(R);
    if (R2 < Q3)
    {
        real_type theta = std::acos(R / std::sqrt(Q3));
        real_type sqrtQ = -2.0 * std::sqrt(Q);
        if (2.0 * theta < M_PI)
            return sqrtQ * std::cos(theta / 3.0);
        else
            return sqrtQ * std::cos((theta + 2.0 * M_PI) / 3.0);
    }
    else
    {
        real_type A = -std::copysign(1.0, R)
                      * std::pow(std::fabs(R) + std::sqrt(R2 - Q3), 1.0 / 3.0);
        real_type B = (A == 0.0) ? 0.0 : Q / A;
        return A + B; /* this is always largest root even if A=B */
    }
}

CELER_FUNCTION real_type Alg1010Solver::calc_phi0(Real4 const& abcd,
                                                  bool scaled) const
{
    /* find phi0 as the dominant root of the depressed and shifted cubic
     * in eq. (79) (see also the discussion in sec. 2.2 of the manuscript) */
    auto [a, b, c, d] = abcd;
    real_type diskr = 9 * ipow<2>(a) - 24 * b;
    /* eq. (87) */
    real_type s;
    if (diskr > 0.0)
    {
        diskr = std::sqrt(diskr);
        s = -2 * b / (3 * a + std::copysign(diskr, a));
    }
    else
    {
        s = -a / 4;
    }
    /* eqs. (83) */
    real_type aq = a + 4 * s;
    real_type bq = b + 3 * s * (a + 2 * s);
    real_type cq = c + s * (2 * b + s * (3 * a + 4 * s));
    real_type dq = d + s * (c + s * (b + s * (a + s)));
    real_type gg = ipow<2>(bq) / 9;
    real_type hh = aq * cq;

    real_type g = hh - 4 * dq - 3 * gg; /* eq. (85) */
    real_type h = (8 * dq + hh - 2 * gg) * bq / 3 - ipow<2>(cq)
                  - dq * ipow<2>(aq); /* eq.
                                  (86)
                                */
    real_type rmax = Alg1010Solver::solve_cubic_analytic_depressed(g, h);
    if (std::isnan(rmax) || std::isinf(rmax))
    {
        rmax = Alg1010Solver::solve_cubic_analytic_depressed_handle_inf(g, h);
        if ((std::isnan(rmax) || std::isinf(rmax)) && scaled)
        {
            // try harder: rescale also the depressed cubic if quartic has been
            // already rescaled
            real_type rfact = Alg1010Solver::CUBIC_RESCAL_FACT;
            real_type rfactsq = ipow<2>(rfact);
            real_type ggss = gg / rfactsq;
            real_type hhss = hh / rfactsq;
            real_type dqss = dq / rfactsq;
            real_type aqs = aq / rfact;
            real_type bqs = bq / rfact;
            real_type cqs = cq / rfact;
            ggss = ipow<2>(bqs) / 9.0;
            hhss = aqs * cqs;
            g = hhss - 4.0 * dqss - 3.0 * ggss;
            h = (8.0 * dqss + hhss - 2.0 * ggss) * bqs / 3
                - cqs * (cqs / rfact) - (dq / rfact) * ipow<2>(aqs);
            rmax = Alg1010Solver::solve_cubic_analytic_depressed(g, h);
            if (std::isnan(rmax) || std::isinf(rmax))
            {
                rmax = Alg1010Solver::solve_cubic_analytic_depressed_handle_inf(
                    g, h);
            }
            rmax *= rfact;
        }
    }
    /* Newton-Raphson used to refine phi0 (see end of sec. 2.2 in the
     * manuscript)
     */
    real_type x = rmax;
    real_type xsq = ipow<2>(x);
    real_type xxx = x * xsq;
    real_type gx = g * x;
    real_type f = x * (xsq + g) + h;
    real_type maxtt = max(std::fabs(xxx), std::fabs(gx));
    if (std::fabs(h) > maxtt)
        maxtt = std::fabs(h);

    if (std::fabs(f) > Alg1010Solver::MACHEPS * maxtt)
    {
        for (int iter = 0; iter < 8; iter++)
        {
            real_type df = 3.0 * xsq + g;
            if (df == 0)
            {
                break;
            }
            real_type xold = x;
            x += -f / df;
            real_type fold = f;
            xsq = ipow<2>(x);
            f = x * (xsq + g) + h;
            if (f == 0)
            {
                break;
            }

            if (std::fabs(f) >= std::fabs(fold))
            {
                x = xold;
                break;
            }
        }
    }
    return x;
}

CELER_FUNCTION real_type Alg1010Solver::calc_err_ldlt(real_type b,
                                                      real_type c,
                                                      real_type d,
                                                      real_type d2,
                                                      real_type l1,
                                                      real_type l2,
                                                      real_type l3) const
{
    /* Eqs. (29) and (30) in the manuscript */
    real_type sum = (b == 0)
                        ? std::fabs(d2 + ipow<2>(l1) + 2.0 * l3)
                        : std::fabs(((d2 + ipow<2>(l1) + 2.0 * l3) - b) / b);
    sum += (c == 0) ? std::fabs(2.0 * d2 * l2 + 2.0 * l1 * l3)
                    : std::fabs(((2.0 * d2 * l2 + 2.0 * l1 * l3) - c) / c);
    sum += (d == 0) ? std::fabs(d2 * ipow<2>(l2) + ipow<2>(l3))
                    : std::fabs(((d2 * ipow<2>(l2) + ipow<2>(l3)) - d) / d);
    return sum;
}

CELER_FUNCTION real_type Alg1010Solver::calc_err_abcd_cmplx(real_type a,
                                                            real_type b,
                                                            real_type c,
                                                            real_type d,
                                                            cmplx_type aq,
                                                            cmplx_type bq,
                                                            cmplx_type cq,
                                                            cmplx_type dq) const
{
    /* Eqs. (68) and (69) in the manuscript for complex alpha1 (aq), beta1
     * (bq), alpha2 (cq) and beta2 (dq) */
    auto cabs = [](cmplx_type comp) { return comp.abs(); };

    real_type sum = (d == 0) ? cabs(bq * dq) : cabs((bq * dq - d) / d);
    sum += (c == 0) ? cabs(bq * cq + aq * dq)
                    : cabs(((bq * cq + aq * dq) - c) / c);
    sum += (b == 0) ? cabs(bq + aq * cq + dq)
                    : cabs(((bq + aq * cq + dq) - b) / b);
    sum += (a == 0) ? cabs(aq + cq) : cabs(((aq + cq) - a) / a);
    return sum;
}

CELER_FUNCTION real_type Alg1010Solver::calc_err_abcd(real_type a,
                                                      real_type b,
                                                      real_type c,
                                                      real_type d,
                                                      real_type aq,
                                                      real_type bq,
                                                      real_type cq,
                                                      real_type dq) const
{
    /* Eqs. (68) and (69) in the manuscript for real alpha1 (aq), beta1 (bq),
     * alpha2 (cq) and beta2 (dq)*/
    real_type sum = (d == 0) ? std::fabs(bq * dq)
                             : std::fabs((bq * dq - d) / d);
    sum += (c == 0) ? std::fabs(bq * cq + aq * dq)
                    : std::fabs(((bq * cq + aq * dq) - c) / c);
    sum += (b == 0) ? std::fabs(bq + aq * cq + dq)
                    : std::fabs(((bq + aq * cq + dq) - b) / b);
    sum += (a == 0) ? std::fabs(aq + cq) : std::fabs(((aq + cq) - a) / a);
    return sum;
}

CELER_FUNCTION real_type Alg1010Solver::calc_err_abc(real_type a,
                                                     real_type b,
                                                     real_type c,
                                                     real_type aq,
                                                     real_type bq,
                                                     real_type cq,
                                                     real_type dq) const
{
    /* Eqs. (48)-(51) in the manuscript */
    real_type sum = (c == 0) ? std::fabs(bq * cq + aq * dq)
                             : std::fabs(((bq * cq + aq * dq) - c) / c);
    sum += (b == 0) ? std::fabs(bq + aq * cq + dq)
                    : std::fabs(((bq + aq * cq + dq) - b) / b);
    sum += (a == 0) ? std::fabs(aq + cq) : std::fabs(((aq + cq) - a) / a);
    return sum;
}

CELER_FUNCTION void Alg1010Solver::NRabcd(real_type a,
                                          real_type b,
                                          real_type c,
                                          real_type d,
                                          real_type* AQ,
                                          real_type* BQ,
                                          real_type* CQ,
                                          real_type* DQ) const
{
    /* Newton-Raphson described in sec. 2.3 of the manuscript for complex
     * coefficients a,b,c,d */
    Real4 xold, x, dx, fvec, vr;
    Array<Array<real_type, 4>, 4> Jinv;
    real_type det;

    x[0] = *AQ;
    x[1] = *BQ;
    x[2] = *CQ;
    x[3] = *DQ;
    vr[0] = d;
    vr[1] = c;
    vr[2] = b;
    vr[3] = a;
    fvec[0] = x[1] * x[3] - d;
    fvec[1] = x[1] * x[2] + x[0] * x[3] - c;
    fvec[2] = x[1] + x[0] * x[2] + x[3] - b;
    fvec[3] = x[0] + x[2] - a;
    real_type errf = 0;
    for (int k1 = 0; k1 < 4; k1++)
    {
        errf += (vr[k1] == 0) ? std::fabs(fvec[k1])
                              : std::fabs(fvec[k1] / vr[k1]);
    }
    for (int iter = 0; iter < 8; iter++)
    {
        real_type x02 = x[0] - x[2];
        det = x[1] * x[1] + x[1] * (-x[2] * x02 - 2.0 * x[3])
              + x[3] * (x[0] * x02 + x[3]);
        if (det == 0.0)
            break;
        Jinv[0][0] = x02;
        Jinv[0][1] = x[3] - x[1];
        Jinv[0][2] = x[1] * x[2] - x[0] * x[3];
        Jinv[0][3] = -x[1] * Jinv[0][1] - x[0] * Jinv[0][2];
        Jinv[1][0] = x[0] * Jinv[0][0] + Jinv[0][1];
        Jinv[1][1] = -x[1] * Jinv[0][0];
        Jinv[1][2] = -x[1] * Jinv[0][1];
        Jinv[1][3] = -x[1] * Jinv[0][2];
        Jinv[2][0] = -Jinv[0][0];
        Jinv[2][1] = -Jinv[0][1];
        Jinv[2][2] = -Jinv[0][2];
        Jinv[2][3] = Jinv[0][2] * x[2] + Jinv[0][1] * x[3];
        Jinv[3][0] = -x[2] * Jinv[0][0] - Jinv[0][1];
        Jinv[3][1] = Jinv[0][0] * x[3];
        Jinv[3][2] = x[3] * Jinv[0][1];
        Jinv[3][3] = x[3] * Jinv[0][2];
        for (int k1 = 0; k1 < 4; k1++)
        {
            dx[k1] = 0;
            for (int k2 = 0; k2 < 4; k2++)
                dx[k1] += Jinv[k1][k2] * fvec[k2];
        }
        for (int k1 = 0; k1 < 4; k1++)
            xold[k1] = x[k1];

        for (int k1 = 0; k1 < 4; k1++)
        {
            x[k1] += -dx[k1] / det;
        }
        fvec[0] = x[1] * x[3] - d;
        fvec[1] = x[1] * x[2] + x[0] * x[3] - c;
        fvec[2] = x[1] + x[0] * x[2] + x[3] - b;
        fvec[3] = x[0] + x[2] - a;
        real_type errfold = errf;
        errf = 0;
        for (int k1 = 0; k1 < 4; k1++)
        {
            errf += (vr[k1] == 0) ? std::fabs(fvec[k1])
                                  : std::fabs(fvec[k1] / vr[k1]);
        }
        if (errf == 0)
            break;
        if (errf >= errfold)
        {
            for (int k1 = 0; k1 < 4; k1++)
                x[k1] = xold[k1];
            break;
        }
    }
    *AQ = x[0];
    *BQ = x[1];
    *CQ = x[2];
    *DQ = x[3];
}

CELER_FUNCTION void
Alg1010Solver::solve_quadratic(real_type a, real_type b, Comp2& roots) const
{
    real_type diskr = ipow<2>(a) - 4 * b;

    if (soft_zero_(diskr))
    {
        roots[0] = cmplx_type(-a / 2, 0.0);
        roots[1] = cmplx_type(no_solution_, no_solution_);
    }
    else if (diskr > 0.0)
    {
        real_type div = -a - std::copysign(std::sqrt(diskr), a);
        real_type zmax = div / 2;
        real_type zmin = (zmax == 0.0) ? 0.0 : b / zmax;

        roots[0] = cmplx_type(zmax, 0.0);
        roots[1] = cmplx_type(zmin, 0.0);
    }
    else
    {
        real_type sqrtd = std::sqrt(-diskr);
        roots[0] = cmplx_type(-a / 2, sqrtd / 2);
        roots[1] = cmplx_type(-a / 2, -sqrtd / 2);
    }
}

CELER_FUNCTION auto Alg1010Solver::operator()(Real5 const& coeff) const
    -> result_type
{
    Comp4 roots;

    cmplx_type acx, bcx, ccx, dcx;
    Array<real_type, 12> l2m, d2m, res;
    Real3 errv, aqv, cqv;
    Array<int, 2> realcase;

    real_type a = coeff[1] / coeff[0];
    real_type b = coeff[2] / coeff[0];
    real_type c = coeff[3] / coeff[0];
    real_type d = coeff[4] / coeff[0];
    real_type phi0 = Alg1010Solver::calc_phi0({a, b, c, d}, 0);

    // simple polynomial rescaling
    real_type rfact = 1.0;
    if (std::isnan(phi0) || std::isinf(phi0))
    {
        rfact = Alg1010Solver::QUART_RESCAL_FACT;
        a /= rfact;
        real_type rfactsq = ipow<2>(rfact);
        b /= rfactsq;
        c /= rfactsq * rfact;
        d /= ipow<2>(rfactsq);
        phi0 = Alg1010Solver::calc_phi0({a, b, c, d}, 1);
    }
    real_type l1 = a / 2; /* eq. (16) */
    real_type l3 = b / 6 + phi0 / 2; /* eq. (18) */
    real_type del2 = c - a * l3; /* defined just after eq. (27) */
    int nsol = 0;
    real_type bl311 = 2. * b / 3. - phi0 - ipow<2>(l1); /* This is d2 as
                                                    defined in eq. (20)*/
    real_type dml3l3 = d - ipow<2>(l3); /* dml3l3 is d3 as defined in eq. (15)
                                    with d2=0 */

    /* Three possible solutions for d2 and l2 (see eqs. (28) and discussion
     * which follows) */
    if (bl311 != 0.0)
    {
        d2m[nsol] = bl311;
        l2m[nsol] = del2 / (2.0 * d2m[nsol]);
        res[nsol] = Alg1010Solver::calc_err_ldlt(
            b, c, d, d2m[nsol], l1, l2m[nsol], l3);
        nsol++;
    }
    if (del2 != 0)
    {
        l2m[nsol] = 2 * dml3l3 / del2;
        if (l2m[nsol] != 0)
        {
            d2m[nsol] = del2 / (2 * l2m[nsol]);
            res[nsol] = Alg1010Solver::calc_err_ldlt(
                b, c, d, d2m[nsol], l1, l2m[nsol], l3);
            nsol++;
        }

        d2m[nsol] = bl311;
        l2m[nsol] = 2.0 * dml3l3 / del2;
        res[nsol] = Alg1010Solver::calc_err_ldlt(
            b, c, d, d2m[nsol], l1, l2m[nsol], l3);
        nsol++;
    }

    real_type l2, d2;
    if (nsol == 0)
    {
        l2 = d2 = 0.0;
    }
    else
    {
        /* we select the (d2,l2) pair which minimizes errors */
        real_type resmin;
        int kmin;
        for (int k1 = 0; k1 < nsol; k1++)
        {
            if (k1 == 0 || res[k1] < resmin)
            {
                resmin = res[k1];
                kmin = k1;
            }
        }
        d2 = d2m[kmin];
        l2 = l2m[kmin];
    }
    int whichcase = 0;
    real_type aq, bq, cq, dq;
    if (d2 < 0.0)
    {
        /* Case I eqs. (37)-(40) */
        real_type gamma = std::sqrt(-d2);
        aq = l1 + gamma;
        bq = l3 + gamma * l2;

        cq = l1 - gamma;
        dq = l3 - gamma * l2;
        if (std::fabs(dq) < std::fabs(bq))
            dq = d / bq;
        else if (std::fabs(dq) > std::fabs(bq))
            bq = d / dq;
        if (std::fabs(aq) < std::fabs(cq))
        {
            nsol = 0;
            if (dq != 0)
            {
                aqv[nsol] = (c - bq * cq) / dq; /* see eqs. (47) */
                errv[nsol] = Alg1010Solver::calc_err_abc(
                    a, b, c, aqv[nsol], bq, cq, dq);
                nsol++;
            }
            if (cq != 0)
            {
                aqv[nsol] = (b - dq - bq) / cq; /* see eqs. (47) */
                errv[nsol] = Alg1010Solver::calc_err_abc(
                    a, b, c, aqv[nsol], bq, cq, dq);
                nsol++;
            }
            aqv[nsol] = a - cq; /* see eqs. (47) */
            errv[nsol]
                = Alg1010Solver::calc_err_abc(a, b, c, aqv[nsol], bq, cq, dq);
            nsol++;
            /* we select the value of aq (i.e. alpha1 in the manuscript) which
             * minimizes errors */
            real_type errmin;
            int kmin;
            for (int k = 0; k < nsol; k++)
            {
                if (k == 0 || errv[k] < errmin)
                {
                    kmin = k;
                    errmin = errv[k];
                }
            }
            aq = aqv[kmin];
        }
        else
        {
            nsol = 0;
            if (bq != 0)
            {
                cqv[nsol] = (c - aq * dq) / bq; /* see eqs. (53) */
                errv[nsol] = Alg1010Solver::calc_err_abc(
                    a, b, c, aq, bq, cqv[nsol], dq);
                nsol++;
            }
            if (aq != 0)
            {
                cqv[nsol] = (b - bq - dq) / aq; /* see eqs. (53) */
                errv[nsol] = Alg1010Solver::calc_err_abc(
                    a, b, c, aq, bq, cqv[nsol], dq);
                nsol++;
            }
            cqv[nsol] = a - aq; /* see eqs. (53) */
            errv[nsol]
                = Alg1010Solver::calc_err_abc(a, b, c, aq, bq, cqv[nsol], dq);
            nsol++;
            /* we select the value of cq (i.e. alpha2 in the manuscript) which
             * minimizes errors */
            real_type errmin;
            int kmin;
            for (int k = 0; k < nsol; k++)
            {
                if (k == 0 || errv[k] < errmin)
                {
                    kmin = k;
                    errmin = errv[k];
                }
            }
            cq = cqv[kmin];
        }
        realcase[0] = 1;
    }
    else if (d2 > 0)
    {
        /* Case II eqs. (53)-(56) */
        real_type gamma = std::sqrt(d2);
        acx = cmplx_type(l1, gamma);
        bcx = cmplx_type(l3, gamma * l2);
        ccx = acx.conj();
        dcx = bcx.conj();
        realcase[0] = 0;
    }
    else
        realcase[0] = -1;  // d2=0
    /* Case III: d2 is 0 or approximately 0 (in this case check which solution
     * is better) */
    if (realcase[0] == -1
        || (std::fabs(d2)
            <= Alg1010Solver::MACHEPS
                   * (std::fabs(2. * b / 3.) + std::fabs(phi0) + ipow<2>(l1))))
    {
        real_type d3 = d - ipow<2>(l3);
        real_type err0 = 0.0;
        if (realcase[0] == 1)
            err0 = Alg1010Solver::calc_err_abcd(a, b, c, d, aq, bq, cq, dq);
        else if (realcase[0] == 0)
            err0 = Alg1010Solver::calc_err_abcd_cmplx(
                a, b, c, d, acx, bcx, ccx, dcx);
        real_type aq1, bq1, cq1, dq1;
        cmplx_type acx1, bcx1, ccx1, dcx1;
        real_type err1 = 0.0;
        if (d3 <= 0)
        {
            realcase[1] = 1;
            aq1 = l1;
            bq1 = l3 + std::sqrt(-d3);
            cq1 = l1;
            dq1 = l3 - std::sqrt(-d3);
            if (std::fabs(dq1) < std::fabs(bq1))
                dq1 = d / bq1;
            else if (std::fabs(dq1) > std::fabs(bq1))
                bq1 = d / dq1;
            err1 = Alg1010Solver::calc_err_abcd(
                a, b, c, d, aq1, bq1, cq1, dq1); /* eq.
                                          (68)
                                        */
        }
        else
        {
            /* complex */
            realcase[1] = 0;
            acx1 = l1;
            bcx1 = cmplx_type(0., std::sqrt(d3)) + l3;
            ccx1 = l1;
            dcx1 = bcx1.conj();
            err1 = Alg1010Solver::calc_err_abcd_cmplx(
                a, b, c, d, acx1, bcx1, ccx1, dcx1);
        }
        if (realcase[0] == -1 || err1 < err0)
        {
            whichcase = 1;  // d2 = 0
            if (realcase[1] == 1)
            {
                aq = aq1;
                bq = bq1;
                cq = cq1;
                dq = dq1;
            }
            else
            {
                acx = acx1;
                bcx = bcx1;
                ccx = ccx1;
                dcx = dcx1;
            }
        }
    }
    if (realcase[whichcase] == 1)
    {
        /* if alpha1, beta1, alpha2 and beta2 are real first refine
         * the coefficient through a Newton-Raphson */
        Alg1010Solver::NRabcd(a, b, c, d, &aq, &bq, &cq, &dq);
        /* finally calculate the roots as roots of p1(x) and p2(x) (see end of
         * sec. 2.1) */
        Comp2 qroots;
        Alg1010Solver::solve_quadratic(aq, bq, qroots);
        roots[0] = qroots[0];
        roots[1] = qroots[1];
        Alg1010Solver::solve_quadratic(cq, dq, qroots);
        roots[2] = qroots[0];
        roots[3] = qroots[1];
    }
    else
    {
        /* complex coefficients of p1 and p2 */
        if (whichcase == 0)
        {  // d2!=0
            auto cdiskr = acx * acx * 0.25 - bcx;
            /* calculate the roots as roots of p1(x) and p2(x) (see end of
             * sec. 2.1)
             */
            auto zx1 = acx * -0.5 + cdiskr.sqrt();
            auto zx2 = acx * -0.5 - cdiskr.sqrt();
            auto zxmax = (zx1.abs() > zx2.abs()) ? zx1 : zx2;
            auto zxmin = bcx / zxmax;
            roots[0] = zxmin;
            roots[1] = zxmin.conj();
            roots[2] = zxmax;
            roots[3] = zxmax.conj();
        }
        else
        {  // d2 ~ 0
            /* never gets here! */
            auto cdiskr = (acx * acx - bcx * 4.0).sqrt();
            auto zx1 = (acx + cdiskr) * -0.5;
            auto zx2 = (acx - cdiskr) * -0.5;
            auto zxmax = (zx1.abs() > zx2.abs()) ? zx1 : zx2;
            auto zxmin = bcx / zxmax;
            roots[0] = zxmax;
            roots[1] = zxmin;
            cdiskr = (ccx * ccx - dcx * 4.0).sqrt();
            zx1 = (ccx + cdiskr) * -0.5;
            zx2 = (ccx - cdiskr) * -0.5;
            zxmax = (zx1.abs() > zx2.abs()) ? zx1 : zx2;
            zxmin = dcx / zxmax;
            roots[2] = zxmax;
            roots[3] = zxmin;
        }
    }
    if (rfact != 1.0)
    {
        for (int k = 0; k < 4; k++)
            roots[k] *= rfact;
    }
    result_type real_roots{
        no_solution_, no_solution_, no_solution_, no_solution_};
    int ri = 0;
    for (int i : range(4))
    {
        cmplx_type new_root = roots[i];

        if (soft_zero_(new_root.imag) && new_root.real != no_solution_
            && new_root.real > 0 && !soft_zero_(new_root.real))
        {
            real_roots[ri] = new_root.real;
            ri += 1;
        }
    }
    return real_roots;
}

CELER_FUNCTION auto Alg1010Solver::operator()(Real4 const& coeff) const
    -> result_type
{
    auto [a, b, c, d] = coeff;
    return (*this)(Real5{a, b, c, d, 0});
}

}  // namespace celeritas

/*
The quartic solver was obtained from the paper: Alberto Giacomo Orellana and
Cristiano De Michele, "Algorithm 1010: Boosting Efficiency in Solving Quartic
Equations with No Compromise in Accuracy," ACM Transactions on Mathematical
Software, 46 (2), pp. 1-28. https://doi.org/10.1145/3386241

OpenMC developers contacted the authors, who have agreed to license their
software under the simplified BSD license, reproduced below:

-------------------------------------------------------------------------------
Copyright (c) 2020 Alberto Giacomo Orellana and Cristiano De Michele
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright notice,
     this list of conditions and the following disclaimer in the documentation
     and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE PYNE DEVELOPMENT TEAM ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
ORs TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
