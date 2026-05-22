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
#include "RoundedNonnegDistribution.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sample from a Poisson distribution using Knuth's algorithm.
 *
 * This algorithm should \em only be used for small, positive mean occurrences
 * since the expected number of samples is proportional to the input value.
 *
 * See the \c PoissonDistribution below for documentation, as it should be used
 * in the "general" case.
 */
template<class RealType = ::celeritas::real_type>
class PoissonDistributionKnuth
{
    static_assert(std::is_floating_point_v<RealType>);

  public:
    //!@{
    //! \name Type aliases
    using real_type = RealType;
    using result_type = ::celeritas::size_type;
    //!@}
  public:
    // Construct with distribution parameter
    explicit inline CELER_FUNCTION PoissonDistributionKnuth(real_type lambda);

    //! Construct with default lambda of 1
    CELER_CEF PoissonDistributionKnuth() : exp_lambda_{constants::euler} {}

    // Sample a random number according to the distribution
    template<class Generator>
    inline CELER_FUNCTION result_type operator()(Generator& rng);

  private:
    real_type exp_lambda_{};
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from the mean of the Poisson distribution.
 */
template<class RealType>
CELER_FUNCTION
PoissonDistributionKnuth<RealType>::PoissonDistributionKnuth(real_type lambda)
    : exp_lambda_{std::exp(lambda)}
{
    CELER_EXPECT(lambda >= 0);
}

//---------------------------------------------------------------------------//
/*!
 * Sample a random number according to the distribution.
 */
template<class RealType>
template<class Generator>
CELER_FUNCTION auto
PoissonDistributionKnuth<RealType>::operator()(Generator& rng) -> result_type
{
    std::make_signed_t<result_type> k{0};
    real_type p = exp_lambda_;
    do
    {
        ++k;
        p *= generate_canonical<real_type>(rng);
    } while (p > 1);
    return static_cast<result_type>(k - 1);
    CELER_ASSERT_UNREACHABLE();
}

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
 * In the degenerate case of \f$ \lambda = 0 \f$, the result is always zero.
 *
 * \note This is effectivelty an inefficient variant combining:
 * - an actual poisson distribution (using Knuth's method),
 * - a rounded nonnegative normal distribution (for \f$ \lambda \gg 1 \f$), and
 * - a delta distribution returning zero (for \f$ \lambda == 0 \f$ ).
 */
template<class RealType = ::celeritas::real_type>
class PoissonDistribution
{
    static_assert(std::is_floating_point_v<RealType>);

  public:
    //!@{
    //! \name Type aliases
    using real_type = RealType;
    using result_type = ::celeritas::size_type;
    //!@}

  public:
    // Construct with distribution parameter
    explicit inline CELER_FUNCTION PoissonDistribution(real_type lambda);

    //! Construct with default lambda of 1
    CELER_FUNCTION PoissonDistribution() : PoissonDistribution{1} {}

    // Sample a random number according to the distribution
    template<class Generator>
    inline CELER_FUNCTION result_type operator()(Generator& rng);

    //! Minimum value of lambda to approximate as a Gaussian
    static CELER_CONSTEXPR_FUNCTION int large_lambda() { return 16; }

  private:
    using PoissonKnuth_t = PoissonDistributionKnuth<real_type>;
    using RoundedNormal_t
        = RoundedNonnegDistribution<NormalDistribution<real_type>, result_type>;

    enum class Method
    {
        zero,
        knuth,
        gaussian
    };
    Method method_;
    PoissonKnuth_t sample_knuth_{};
    RoundedNormal_t sample_normal_{};
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
    else if (lambda <= PoissonDistribution::large_lambda())
    {
        method_ = Method::knuth;
        sample_knuth_ = PoissonDistributionKnuth{lambda};
    }
    else
    {
        method_ = Method::gaussian;
        sample_normal_ = RoundedNormal_t{lambda, std::sqrt(lambda)};
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
        case Method::knuth:
            return sample_knuth_(rng);
        case Method::gaussian:
            // Use Gaussian approximation rounded to nearest nonneg integer
            return sample_normal_(rng);
    };
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
