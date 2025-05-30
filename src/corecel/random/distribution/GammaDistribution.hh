//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/distribution/GammaDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/Constants.hh"

#include "NormalDistribution.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sample from a gamma distribution.
 *
 * The gamma distribution can be parameterized with a shape parameter \f$
 * \alpha \f$ and a scale parameter \f$ \beta \f$ and has the PDF:
 * \f[
   f(x; \alpha, \beta) = \frac{x^{\alpha - 1} e^{-x / \beta}}{\beta^\alpha
 \Gamma(\alpha)} \quad \mathrm{for}\  x > 0, \quad \alpha, \beta > 0
   \f]
 * The algorithm described in \citet{marsaglia-gamma-2000,
 * https://dl.acm.org/doi/10.1145/358407.358414} is used here to
 * generate gamma-distributed random variables. The steps are:
 *  1. Set \f$ d = \alpha - 1/3 \f$ and \f$ c = 1 / \sqrt{9d} \f$
 *  2. Generate random variates \f$ Z \sim N(0,1) \f$ and \f$ U \sim U(0,1) \f$
 *  3. Set \f$ v = (1 + cZ)^3 \f$
 *  4. If \f$ \log U < Z^2 / 2 + d(1 - v + \log v) \f$ return \f$ dv \f$.
 *     Otherwise, go to step 2.
 * A squeeze function can be used to avoid the two logarithms in most cases by
 * accepting early if \f$ U < 1 - 0.0331 Z^4 \f$.
 *
 * Though this method is valid for \f$ \alpha \ge 1 \f$, it can easily be
 * extended for \f$ \alpha < 1 \f$: if \f$ X \sim \Gamma(\alpha + 1) \f$
 * and \f$ U \sim U(0,1) \f$, then \f$ X U^{1/\alpha} \sim \Gamma(\alpha) \f$.
 */
template<class RealType = ::celeritas::real_type>
class GammaDistribution
{
  public:
    //!@{
    //! \name Type aliases
    using real_type = RealType;
    using result_type = real_type;
    //!@}

  public:
    // Construct with shape and scale parameters
    explicit inline CELER_FUNCTION
    GammaDistribution(real_type alpha = 1, real_type beta = 1);

    // Sample a random number according to the distribution
    template<class Generator>
    inline CELER_FUNCTION result_type operator()(Generator& rng);

  private:
    real_type const alpha_;
    real_type const beta_;
    real_type const alpha_p_;
    real_type const d_;
    real_type const c_;
    NormalDistribution<real_type> sample_normal_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from the shape and scale parameters.
 */
template<class RealType>
CELER_FUNCTION
GammaDistribution<RealType>::GammaDistribution(real_type alpha, real_type beta)
    : alpha_(alpha)
    , beta_(beta)
    , alpha_p_(alpha < 1 ? alpha + 1 : alpha)
    , d_(alpha_p_ - real_type(1) / 3)
    , c_(celeritas::rsqrt(9 * d_))
{
    CELER_EXPECT(alpha_ > 0);
    CELER_EXPECT(beta_ > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Sample a random number according to the distribution.
 */
template<class RealType>
template<class Generator>
CELER_FUNCTION auto GammaDistribution<RealType>::operator()(Generator& rng)
    -> result_type
{
    real_type u, v, z;
    do
    {
        do
        {
            z = sample_normal_(rng);
            v = 1 + c_ * z;
        } while (v <= 0);
        v = ipow<3>(v);
        u = generate_canonical<real_type>(rng);
    } while (u > 1 - real_type(0.0331) * ipow<4>(z)
             && std::log(u) > real_type(0.5) * ipow<2>(z)
                                  + d_ * (1 - v + std::log(v)));

    result_type result = d_ * v * beta_;
    if (alpha_ != alpha_p_)
        result *= fastpow(generate_canonical<real_type>(rng), 1 / alpha_);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
