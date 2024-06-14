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
    int max_depth{30};  //!< Maximum number of outer iterations
};

//---------------------------------------------------------------------------//
/*!
 * Perform numerical integration of a generic 1-D function.
 *
 * Currently this is a very simple Newton-Coates integrator extracted from
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
 * Calculate the integral over the given dx.
 */
template<class F>
auto Integrator<F>::operator()(argument_type lo, argument_type hi) -> result_type
{
    constexpr real_type half{0.5};

    real_type delta = hi - lo;
    real_type result = half * delta * (eval_(lo) + eval_(hi));

    size_type points = 1;
    real_type dx = delta;

    int remaining_trials = max_depth_;
    do
    {
        // Accumulate the sum of midpoints along the grid spacing
        real_type accum = 0;
        for (size_type i = 0; i < points; ++i)
        {
            real_type x = std::fma((half + static_cast<real_type>(i)), dx, lo);
            accum += eval_(x);
        }

        // Average previous and new integrations, i.e. combining all the
        // existing and current grid points
        real_type prev = result;
        result = half * (prev + accum * dx);
        if (std::fabs(result - prev) < epsilon_ * std::fabs(result))
        {
            return result;
        }
        points *= 2;
        dx *= half;
    } while (--remaining_trials > 0);

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
