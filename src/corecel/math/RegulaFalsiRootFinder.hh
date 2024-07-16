//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
/*! \file corecel/math/RegulaFalsiRootFinder.hh
 *
 * Perform a Regula Falsi Iteration given a root function \em func and
 * toolerance \em tol .
 *
 * Using a \em left and \em right bound a Regula Falsi approximates the \em
 * root as: \f[ root = (left * func(right) - right * func(left)) / (func(right)
 * - func(left)) \f]
 *
 * Then value of \em func at the root is calculated compared to values of
 * \em func at the bounds. The bound which has the same sign \em func(root)
 */
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
template<class F>
class RegulaFalsi
{
  public:
    // Contructpr of Regula Falsi
    inline CELER_FUNCTION RegulaFalsi(F&& func, real_type tol);

    // Solve for a root between two points
    real_type operator()(real_type left, real_type right);

  private:
    F func_;
    real_type tol_;
};

//---------------------------------------------------------------------------//
// TEMPLATE DEDUCTION
//---------------------------------------------------------------------------//

template<class F, class... Args>
RegulaFalsi(F&&, Args...) -> RegulaFalsi<F>;

//---------------------------------------------------------------------------//
/*!
 * Construct from function.
 */
template<class F>
CELER_FUNCTION RegulaFalsi<F>::RegulaFalsi(F&& func, real_type tol)
    : func_{celeritas::forward<F>(func)}, tol_{tol}
{
    CELER_EXPECT(tol > 0);
}

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
/*!
 * Perform Regula Falsi in defined bounds \em left and \em right with
 * parameters defined in \em params.
 */
template<class F>
CELER_FUNCTION real_type RegulaFalsi<F>::operator()(real_type left,
                                                    real_type right)
{
    // Initialize Iteration parameters
    real_type f_left = func_(left);
    real_type f_right = func_(right);
    real_type f_root = 1;
    real_type root = 0;
    int i = 0;

    // Iterate on root
    do
    {
        // Calcuate root
        root = (left * f_right - right * f_left) / (f_right - f_left);

        // Root function value of root
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
        i++;
    } while (std::fabs(f_root) > tol_ && i <= 100);

    return root;
}

}  // namespace celeritas