//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/IllinoisRootFinder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>
#include <type_traits>

#include "celeritas_config.h"
#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"

#include "detail/MathImpl.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Perform Regula Falsi (see RegulaFalsi for more details) iterations given a
 * root function \em func and tolerance \em tol using the Illinois method.
 *
 * Illonois method modifies the standard approach by comparing the sign of
 * \em func(root) approximation in the current iteration with the previous
 * approximation. If both iterations are on the same side then the \em func at
 * the bound on the other side is halved.
 */
template<class F>
class Illinois
{
  public:
    // Contruct with function to solve and solution tolerance
    inline CELER_FUNCTION Illinois(F&& func, real_type tol);

    // Solve for a root between two points
    inline real_type operator()(real_type left, real_type right);

  private:
    F func_;
    real_type tol_;

    // Maximum amount of iterations
    static constexpr inline int max_iters_ = 30;
};

//---------------------------------------------------------------------------//
// TEMPLATE DEDUCTION
//---------------------------------------------------------------------------//

template<class F, class... Args>
Illinois(F&&, Args...) -> Illinois<F>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from function.
 */
template<class F>
CELER_FUNCTION Illinois<F>::Illinois(F&& func, real_type tol)
    : func_{celeritas::forward<F>(func)}, tol_{tol}
{
    CELER_EXPECT(tol_ > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Solve for a root between the two points.
 */
template<class F>
CELER_FUNCTION real_type Illinois<F>::operator()(real_type left,
                                                 real_type right)
{
    // Initialize Iteration parameters
    real_type f_left = func_(left);
    real_type f_right = func_(right);
    real_type f_root = 1;
    real_type root = 0;
    real_type side = 0;
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
            if (side == -1)
            {
                f_right *= 0.5;
            }
            side = -1;
        }
        else
        {
            right = root;
            f_right = f_root;
            if (side == 1)
            {
                f_left *= 0.5;
            }
            side = 1;
        }
    } while (std::fabs(f_root) > tol_ && --remaining_iters > 0);

    CELER_ENSURE(remaining_iters > 0);
    return root;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
