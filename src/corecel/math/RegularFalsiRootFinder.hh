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
#include <functional>
#include <type_traits>

#include "celeritas_config.h"
#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Span.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArrayUtils.hh"
#include "orange/OrangeTypes.hh"

#include "detail/MathImpl.hh"

namespace celeritas
{
template<class F>
class RegularFalsi
{
  public:
    //@{
    //! \name Type aliases
    using Real4 = Array<real_type, 4>;
    //@}

  public:
    // Contructpr of Regular Falsi
    inline CELER_FUNCTION RegularFalsi(F&& func, real_type tol);

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
template<class F>
CELER_FUNCTION RegularFalsi<F>::RegularFalsi(F&& func, real_type tol)
    : func_{func}, tol_{tol}
{
    CELER_EXPECT(tol > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Perform Regular Falsi in defined bounds \em left and \em right with
 * parameters defined in \em params.
 */
template<class F>
CELER_FUNCTION real_type RegularFalsi<F>::operator()(real_type left,
                                                     real_type right,
                                                     Real4 params)
{
    // Initialize Iteration parameters
    real_type f_left = func_(left, params);
    real_type f_right = func_(right, params);
    real_type f_root = 1;
    real_type root = 0;

    // Iterate on root
    while (f_root > tol_)
    {
        // Calcuate root
        root = (left * f_right - right * f_left) / (f_right - f_left);

        // Root function value of root
        f_root = func_(root, params);

        // Update bounds with iterated root
        if ((0 < f_left) - (f_left < 0) == (0 < f_root) - (f_root < 0))
        {
            left = root;
            f_left = f_root;
        }
        else
        {
            right = root;
            f_right = f_root;
        }
    }
    return root;
}

}  // namespace celeritas