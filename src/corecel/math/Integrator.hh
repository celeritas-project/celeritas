//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/Integrator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"

#include "Algorithms.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Solver options
struct IntegratorOptions
{
    real_type epsilon{1e-3};  //!< Convergence criterion
    int max_depth{1000};  //!< Maximum number of divisions for integrating
};

//---------------------------------------------------------------------------//
/*!
 * Perform numerical integration of a generic 1-D function.
 *
 * Currently this is a very simple Newton-Coates-like integrator extracted from
 * NuclearZoneBuilder. It should be improved for robustness, accuracy, and
 * efficiency, probably by using a Gauss-Legendre quadrature.
 *
 * This class is \em only to be used during setup.
 */
template<class F>
class Integrator
{
  public:
    //!@{
    //! \name Type aliases
    using argument_type = real_type;
    using result_type = real_type;
    using Options = IntegratorOptions;
    //!@}

  public:
    // Construct with the function and options
    inline Integrator(F&& func, Options opts);

    //! Construct with the function and default options
    explicit Integrator(F&& func) : Integrator{std::forward<F>(func), {}} {}

    // Calculate the integral over the given integral
    inline result_type operator()(argument_type lo, argument_type hi);

  private:
    F eval_;
    real_type epsilon_;
    int max_depth_;
};

//---------------------------------------------------------------------------//
// TEMPLATE DEDUCTION
//---------------------------------------------------------------------------//

template<class F, class... Args>
Integrator(F&&, Args...) -> Integrator<F>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with the function and options.
 */
template<class F>
Integrator<F>::Integrator(F&& func, Options opts)
    : eval_{std::forward<F>(func)}
    , epsilon_{opts.epsilon}
    , max_depth_{opts.max_depth}
{
    CELER_EXPECT(epsilon_ > 0);
    CELER_EXPECT(max_depth_ > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the integral over the given interval.
 */
template<class F>
auto Integrator<F>::operator()(argument_type lo, argument_type hi) -> result_type
{
    constexpr real_type half{0.5};

    real_type delta = hi - lo;
    real_type prev = half * delta * (eval_(lo) + eval_(hi));

    int depth = 1;
    real_type interval = delta;
    real_type result = 0;

    bool succeeded = false;
    int remaining_trials = max_depth_;

    do
    {
        delta *= half;

        real_type x = lo - delta;
        real_type fi = 0;

        for (int i = 0; i < depth; ++i)
        {
            x += interval;
            fi += eval_(x);
        }

        result = half * prev + fi * delta;

        if (std::fabs(result - prev) < epsilon_ * std::fabs(result))
        {
            succeeded = true;
        }
        else
        {
            depth *= 2;
            interval = delta;
            prev = result;
        }
    } while (!succeeded && --remaining_trials > 0);

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
