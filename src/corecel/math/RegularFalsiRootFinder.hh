//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
/*! \file corecel/math/RegularFalsiRootFinder.hh
 *
 * Perform a Regular Falsi Iteration given a root function \em func and
 * toolerance \em tol .
 *
 * Using a \em left and \em right bound a Regular Falsi approximates the \em
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

#include "detail/MathImpl.hh"

namespace celeritas
{
template<class F>
class RegularFalsiRootFinder
{
  public:
    //@{
    //! \name Type aliases
    using Real4 = Array<real_type, 4>;
    //@}

  public:
    // Contructpr of Regular Falsi
    CELER_FUNCTION inline RegulaFalsiRootFinder(F&& func, real_type tol);

    // Solve between
    real_type operator()(real_type left, real_type right, Real4 params);

  private:
    F&& func_;
    real_type tol_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
/*!
 * Construct from function.
 */
CELER_FUNCTION RegularFalsiRootFinder(F&& func, real_type tol)
    : func_{func}, tol_{tol}
{
    CELER_EXPECT(tol > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Perform Regular Falsi in defined bounds \em left and \em right with
 * parameters defined in \em params.
 */
CELER_FUNCTION real_type operator()(real_type left,
                                    real_type right,
                                    Real4 params)
{
    real_type f_left = func_(left, params);
    real_type f_right = func_(right, params);
    real_type f_root = 1;
    real_type root;

    while (f_root > tol_)
    {
        root = (left * func_right - right * func_left)
               / (func_right - func_left);
        f_root = func_(root, params)
    }
    return root
}

}  // namespace celeritas