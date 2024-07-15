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
    // Contructpr of Regular Falsi
    CELER_FUNCTION inline RegulaFalsiRootFinder(F&& func, real_type tol);

    // Solve between
    real_type operator()(real_type left, real_type right);

  private:
    F&& func_;
    real_type tol_;
};

CELER_FUNCTION RegularFalsiRootFinder(F&& func, real_type tol)
    : func_{func}, tol_{tol}
{
    CELER_EXPECT(tol > 0);
}

}  // namespace celeritas