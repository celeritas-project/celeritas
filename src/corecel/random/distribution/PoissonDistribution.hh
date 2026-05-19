//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/distribution/PoissonDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"

#include "NormalDistribution.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sample from a generalized Poisson distribution.
 *
 * The Poisson distribution describes the probability of \f$ k \f$ events
 * occurring in a fixed interval given a mean rate of occurrence \f$ \lambda
 \f$
 * and has the PMF:
 * \f[
   f(k; \lambda) = \frac{\lambda^k e^{-\lambda}}{k!} \:.
   \f]
 * For small \f$ \lambda \f$, a direct method described in
 * \cite{knuth-artcomputer-1968} can be
 * used to generate samples from the Poisson distribution. Uniformly
 * distributed random numbers are generated until the relation
 * \f[
   \prod_{k = 1}^n U_k \le e^{-\lambda}
   \f]
 * is satisfied; then, the random variable \f$ X = n - 1 \f$ will have a
 * Poisson distribution. On average this approach requires the generation of
 * \f$ \lambda + 1 \f$ uniform random samples, so a different method should be
 * used for large \f$ \lambda \f$.
 *
 * Geant4 uses Knuth's algorithm for \f$ \lambda \le 16 \f$ and a Gaussian
 * approximation for \f$ \lambda > 16 \f$ (see \c G4Poisson), which is faster
 * but less accurate than other methods. The same approach is used here.
 *
 * In the degenerate case of \f$ \lambda = 0 \f$, the result is always zero.
 *
 * \todo Rename to GeneralPoissonDistribution (or something similar) since
 * it's effectively an inefficient variant combining:
 * - an actual poisson distribution,
 * - an "integer normal" distribution, and
 * - a "zero" distribution.
 */
template<class RealType = ::celeritas::real_type>
class PoissonDistribution
{
  public:
    //!@{
    //! \name Type aliases
    using real_type = RealType;
    using result_type = unsigned int;
    //!@}

  public:
    // Construct with distribution parameter
    explicit inline CELER_FUNCTION PoissonDistribution(real_type lambda);

    //! Construct with default lambda of 1
    CELER_FUNCTION PoissonDistribution() : PoissonDistribution{1} {}

    // Sample a random number according to the distribution
    template<class Generator>
    inline CELER_FUNCTION result_type operator()(Generator& rng);

    //! Maximum value of lambda for using the direct method
    static CELER_CONSTEXPR_FUNCTION int lambda_threshold() { return 16; }

  private:
    enum class Method
    {
        zero,
        poisson,
        gaussian
    };
    Method method_;
    real_type exp_lambda_{};
    NormalDistribution<real_type> sample_normal_{};
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from the mean of the Poisson distribution.
 */
template<class RealType>
CELER_FUNCTION
PoissonDistribution<RealType>::PoissonDistribution(real_type lambda)
{
    CELER_EXPECT(lambda >= 0);

    if (lambda <= 0)
    {
        method_ = Method::zero;
    }
    else if (lambda <= PoissonDistribution::lambda_threshold())
    {
        method_ = Method::poisson;
        exp_lambda_ = std::exp(lambda);
    }
    else
    {
        method_ = Method::gaussian;
        // Add 0.5 to mean for correct rounding
        sample_normal_ = NormalDistribution{lambda + 0.5, std::sqrt(lambda)};
    }
}

//---------------------------------------------------------------------------//
/*!
 * Sample a random number according to the distribution.
 */
template<class RealType>
template<class Generator>
CELER_FUNCTION auto PoissonDistribution<RealType>::operator()(Generator& rng)
    -> result_type
{
    switch (method_)
    {
        case Method::zero:
            return 0;
        case Method::poisson: {
            int k = 0;
            real_type p = exp_lambda_;
            do
            {
                ++k;
                p *= generate_canonical<real_type>(rng);
            } while (p > 1);
            return static_cast<result_type>(k - 1);
        }
        case Method::gaussian:
            // Use Gaussian approximation rounded to nearest integer
            return static_cast<result_type>(
                clamp_to_nonneg(sample_normal_(rng)));
    };
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
