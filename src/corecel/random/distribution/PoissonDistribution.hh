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
#include "corecel/math/Algorithms.hh"
#include "celeritas/Constants.hh"

#include "NormalDistribution.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sample from a Poisson distribution.
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
 * \todo Break this into two distributions: one actual poisson distribution,
 * one "integer normal" distribution, and a variant type that selects between
 * them. In most cases we care about, lambda is small.
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
    // Construct with defaults
    explicit inline CELER_FUNCTION PoissonDistribution(real_type lambda = 1);

    // Sample a random number according to the distribution
    template<class Generator>
    inline CELER_FUNCTION result_type operator()(Generator& rng);

    //! Maximum value of lambda for using the direct method
    static CELER_CONSTEXPR_FUNCTION int lambda_threshold() { return 16; }

  private:
    real_type const lambda_;
    NormalDistribution<real_type> sample_normal_;
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
    : lambda_(lambda), sample_normal_(lambda_, std::sqrt(lambda_))
{
    CELER_EXPECT(lambda_ > 0);
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
    if (lambda_ <= PoissonDistribution::lambda_threshold())
    {
        // Use direct method
        int k = 0;
        real_type p = std::exp(lambda_);
        do
        {
            ++k;
            p *= generate_canonical<real_type>(rng);
        } while (p > 1);
        return static_cast<result_type>(k - 1);
    }
    // Use Gaussian approximation rounded to nearest integer
    return result_type(sample_normal_(rng) + real_type(0.5));
}
//---------------------------------------------------------------------------//
}  // namespace celeritas
