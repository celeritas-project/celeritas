//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/RegulaFalsiRootFinder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>
#include <type_traits>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"

#include "Algorithms.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Find a local root using the Regula Falsi algorithm.
 *
 * Using left bound \f$ x_l \f$ and right bound  \f$ x_r \f$, Regula Falsi
 * approximates the root \em x' as \f[
 * x' = \frac{x_l * f(x_r) - x_r * f(x_l)}{f(x_r) f(x_l)}
 * \f]
 *
 * Then value of \em f at the root is calculated compared to values of
 * \em f at the bounds. The root is then used to update the bounds based on
 * the sign of \f$ f(x') \f$ and whether it matches the sign of \f$ f(x_l) \f$
 * or \f$ f(x_r) \f$ .
 */
template<class F>
class RegulaFalsiRootFinder
{
  public:
    // Construct with function to solve and solution tolerance
    inline CELER_FUNCTION RegulaFalsiRootFinder(F&& func, real_type tol);

    // Solve for a root between two points
    inline CELER_FUNCTION real_type operator()(real_type left, real_type right);

  private:
    F func_;
    real_type tol_;

    // Maximum amount of iterations
    static constexpr int max_iters_ = 50;
};

//---------------------------------------------------------------------------//
// TEMPLATE DEDUCTION
//---------------------------------------------------------------------------//

template<class F, class... Args>
CELER_FUNCTION RegulaFalsiRootFinder(F&&, Args...) -> RegulaFalsiRootFinder<F>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from function.
 */
template<class F>
CELER_FUNCTION
RegulaFalsiRootFinder<F>::RegulaFalsiRootFinder(F&& func, real_type tol)
    : func_{celeritas::forward<F>(func)}, tol_{tol}
{
    CELER_EXPECT(tol_ > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Solve for a root between the two points.
 */
template<class F>
CELER_FUNCTION real_type RegulaFalsiRootFinder<F>::operator()(real_type left,
                                                              real_type right)
{
    // Initialize Iteration parameters
    real_type f_left = func_(left);
    real_type f_right = func_(right);
    real_type f_root = 1;
    real_type root = 0;
    int remaining_iters = max_iters_;

    // Iterate on root
    do
    {
        // Estimate root and update value
        root = (left * f_right - right * f_left) / (f_right - f_left);
        f_root = func_(root);

        // Update the bound which produces the same sign as the root
        if (signum(f_left) == signum(f_root))
        {
            left = root;
            f_left = f_root;
        }
        else
        {
            right = root;
            f_right = f_root;
        }
    } while (std::fabs(f_root) > tol_ && --remaining_iters > 0);

    CELER_ENSURE(remaining_iters > 0);
    return root;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
