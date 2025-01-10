//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/BisectionRootFinder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>
#include <type_traits>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Perform Bisection iterations given a root function \em func and
 * tolerance \em tol .
 *
 * Using a \em left and \em right bound a Bisection approximates the \em
 * root as: \f[ root = 0.5 * (left + right) \f]
 *
 * Then value of \em func at the root is calculated compared to values of
 * \em func at the bounds. The \em root is then used update the bounds based on
 * the sign of \em func(root) and whether it matches the sign of \em func(left)
 * or \em func(right) . Performing this update of the bounds allows for the
 * iteration on root, using the convergence criteria based on \em func(root)
 * proximity to 0.
 */
template<class F>
class BisectionRootFinder
{
  public:
    // Contruct with function to solve and solution tolerance
    inline CELER_FUNCTION BisectionRootFinder(F&& func, real_type tol);

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
CELER_FUNCTION BisectionRootFinder(F&&, Args...) -> BisectionRootFinder<F>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from function.
 */
template<class F>
CELER_FUNCTION
BisectionRootFinder<F>::BisectionRootFinder(F&& func, real_type tol)
    : func_{celeritas::forward<F>(func)}, tol_{tol}
{
    CELER_EXPECT(tol_ > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Solve for a root between the two points.
 */
template<class F>
CELER_FUNCTION real_type BisectionRootFinder<F>::operator()(real_type left,
                                                            real_type right)
{
    // Initialize Iteration parameters
    real_type f_left = func_(left);
    real_type f_root = 1;
    real_type root = 0;
    int remaining_iters = max_iters_;

    // Iterate on root
    do
    {
        // Estimate root and update value
        root = real_type(0.5) * (left + right);
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
        }
    } while (std::fabs(f_root) > tol_ && --remaining_iters > 0);

    CELER_ENSURE(remaining_iters > 0);
    return root;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
