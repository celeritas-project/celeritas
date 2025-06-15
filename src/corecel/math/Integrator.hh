//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/Integrator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"

#include "Algorithms.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Solver options
struct IntegratorOptions
{
    using DepthInt = short unsigned int;

    real_type epsilon{1e-3};  //!< Convergence criterion
    DepthInt max_depth{20};  //!< Maximum number of outer iterations

    //! Whether the options are valid
    explicit operator bool() const
    {
        return epsilon > 0 && max_depth > 0
               && static_cast<std::size_t>(max_depth) < 8 * sizeof(size_type);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Perform numerical integration of a generic 1-D function.
 *
 * Currently this is a simple nonrecursive Newton-Coates integrator extracted
 * from NuclearZoneBuilder. It should be improved for robustness, accuracy, and
 * efficiency, probably by using a Gauss-Legendre quadrature.
 *
 * This class is to be used \em only during setup.
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
    Options options_;

    //! Whether convergence is achieved (simple soft equality)
    bool converged(real_type prev, real_type cur) const
    {
        CELER_EXPECT(!std::isinf(cur) && !std::isnan(cur));
        return std::fabs(cur - prev) <= options_.epsilon * std::fabs(cur);
    }
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
    : eval_{std::forward<F>(func)}, options_{opts}
{
    CELER_EXPECT(options_);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the integral over the given dx.
 */
template<class F>
auto Integrator<F>::operator()(argument_type lo, argument_type hi)
    -> result_type
{
    CELER_EXPECT(lo < hi);
    constexpr real_type half{0.5};

    size_type num_points = 1;
    real_type dx = (hi - lo);

    // Initial estimate is the endpoints of the function
    real_type result = half * dx * (eval_(lo) + eval_(hi));
    real_type prev{};

    auto remaining_trials = options_.max_depth;
    do
    {
        // Accumulate the sum of midpoints along the grid spacing
        real_type accum = 0;
        for (auto i : range(num_points))
        {
            real_type x = std::fma((half + static_cast<real_type>(i)), dx, lo);
            accum += eval_(x);
        }

        // Average previous and new integrations, i.e. combining all the
        // existing and current grid points
        prev = result;
        result = half * (prev + accum * dx);

        // Increase number of intervals (and decrease dx) for next iteration
        num_points *= 2;
        dx *= half;
    } while (!this->converged(prev, result) && --remaining_trials > 0);

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
