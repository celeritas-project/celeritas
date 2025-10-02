//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/IllinoisRootFinder.hh
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
 * Solve for a single local root using the Illinois method.
 *
 * Perform Regula Falsi (see celeritas::RegulaFalsiSolver for more details)
 * iterations given a root function \em func and tolerance \em tol using the
 * Illinois method.
 *
 * Illinois method modifies the standard approach by comparing the sign of
 * \em func(root) approximation in the current iteration with the previous
 * approximation. If both iterations are on the same side then the \em func at
 * the bound on the other side is halved.
 */
template<class F>
class IllinoisRootFinder
{
  public:
    // Construct with function to solve and solution tolerance
    inline CELER_FUNCTION IllinoisRootFinder(F&& func, real_type tol);

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
CELER_FUNCTION IllinoisRootFinder(F&&, Args...) -> IllinoisRootFinder<F>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from function.
 */
template<class F>
CELER_FUNCTION
IllinoisRootFinder<F>::IllinoisRootFinder(F&& func, real_type tol)
    : func_{celeritas::forward<F>(func)}, tol_{tol}
{
    CELER_EXPECT(tol_ > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Solve for a root between the two points.
 */
template<class F>
CELER_FUNCTION real_type IllinoisRootFinder<F>::operator()(real_type left,
                                                           real_type right)
{
    //! Enum defining side of approximated root to true root
    enum class Side
    {
        left = -1,
        init = 0,
        right = 1
    };

    // Initialize Iteration parameters
    real_type f_left = func_(left);
    real_type f_right = func_(right);
    real_type f_root = 1;
    real_type root = 0;
    Side side = Side::init;
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
            if (side == Side::left)
            {
                f_right *= real_type(0.5);
            }
            side = Side::left;
        }
        else
        {
            right = root;
            f_right = f_root;
            if (side == Side::right)
            {
                f_left *= real_type(0.5);
            }
            side = Side::right;
        }
    } while (std::fabs(f_root) > tol_ && --remaining_iters > 0);

    CELER_ENSURE(remaining_iters > 0);
    return root;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
