//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDmu-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/distribution/UrbanLargeAngleDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/random/distribution/BernoulliDistribution.hh"
#include "corecel/random/distribution/PowerDistribution.hh"
#include "corecel/random/distribution/UniformRealDistribution.hh"
#include "corecel/random/engine/CachedRngEngine.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sample the large-angle MSC scattering cosine.
 *
 * \citet{urban-msc-2006,
 * https://cds.cern.ch/record/1004190/} proposes a convex combination of three
 * probability distribution functions:
 * \f[
 * \begin{aligned}
 *  g_0(\mu) &\sim \exp(-a(1 - \mu)), \\
 *  g_1(\mu) &\sim (b - \mu)^{-d}, \\
 *  g_2(\mu) &\sim 1
 * \end{aligned}
 * \f]
 * which have normalizing constants and sum to
 * \f[
 * g(\mu) = p_1 p_2 g_0(\mu) + p_1(1-p_2) g_1(\mu) + (1-p_1) g_2(\mu).
 * \f]
 *
 * In this distribution for large angles, \f$ p_2 = 1 \f$ so only the
 * exponential and constant terms are sampled.
 *
 *
 * The Goudsmit-Saunderson moments for the expected angular deflection
 * \f$ \theta \f$ over a physical path length \f$ s \f$ are: \f[
    \langle \cos \theta \rangle
     \equiv \langle \mu \rangle
     = \ee^{-s/\lambda_1} \ ,
 * \f] and \f[
    \langle \cos^2 \theta \rangle
      \equiv \langle \mu^2 \rangle
      = \frac{1}{3}\left(1 + 2 \ee^{-s / \lambda_2}\right) \ ,
 * \f]
 * where the transport mean free paths \f$ \lambda_l \f$ are related to the
 * \f$ l \f$th angular moment of the elastic cross section scattering  (see
 * Eqs. 6-8, 15-16 from \citet{fernandez-msc-1993,
 * https://doi.org/10.1016/0168-583X(93)95827-R}
 * ).
 *
 * Given the number of mean free paths \f[
    \tau \equiv \frac{s}{\lambda_1} \ ,
 * \f]
 * and from \citet{kawrakow-condensedhistory-1998,
 * https://doi.org/10.1016/S0168-583X(98)00274-2} that for kinetic energies
 * between a few keV and infinity, \f[
   2 < \frac{\lambda_2}{\lambda_1} < \infty \ ,
 * \f]
 * this class calculates the mean scattering angle and approximates the second
 * moment of the scattering cosine using
 * \f$ \lambda_2 \approx 2.5 \lambda_1 \f$.
 *
 * Using these moments, Urban calculates: \f[
   a = \frac{2\langle \mu \rangle + 9\langle \mu^2 \rangle - 3}
            {2\langle \mu \rangle - 3\langle \mu^2 \rangle + 1}
 * \f] and
 * \f[
   p_1 = \frac{(a + 2)\langle \mu \rangle}{a} \,.
 * \f]
 */
class UrbanLargeAngleDistribution
{
  public:
    // Construct with mean free path tau
    explicit inline CELER_FUNCTION UrbanLargeAngleDistribution(real_type tau);

    // Sample cos(theta)
    template<class Engine>
    inline CELER_FUNCTION real_type operator()(Engine& rng);

  private:
    BernoulliDistribution select_pow_{0};
    PowerDistribution<> sample_pow_{0};
    UniformRealDistribution<> sample_uniform_{};
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with mean values.
 */
CELER_FUNCTION
UrbanLargeAngleDistribution::UrbanLargeAngleDistribution(real_type tau)
{
    CELER_EXPECT(tau > 0);
    // Eq. 8.2 and \f$ \cos^2\theta \f$ term in Eq. 8.3 in PRM
    real_type mumean = std::exp(-tau);
    // NOTE: tau_big = 8 -> ~0.0003 < mumean < 1

    real_type musqmean = (1 + 2 * std::exp(real_type(-2.5) * tau)) / 3;
    real_type a = (2 * mumean + 9 * musqmean - 3)
                  / (2 * mumean - 3 * musqmean + 1);

    select_pow_ = BernoulliDistribution{mumean * (1 + 2 / a)};
    sample_pow_ = PowerDistribution<>{a};
}

//---------------------------------------------------------------------------//
/*!
 * Sample from two parameters of the model function.
 *
 * \todo The cached RNG value is not necessary and is only for reproducing
 * previous results. It should be deleted the next time a rebaselining is
 * performed.
 */
template<class Engine>
CELER_FUNCTION real_type UrbanLargeAngleDistribution::operator()(Engine& rng)
{
    // Save the RNG result to preserve RNG stream from older Celeritas
    auto temp_rng = cache_rng_values<real_type, 1>(rng);
    // Sample u = (cos \theta + 1)/ 2
    real_type half_angle = select_pow_(rng) ? sample_pow_(temp_rng)
                                            : sample_uniform_(temp_rng);
    CELER_ASSERT(half_angle >= 0 && half_angle <= 1);

    // Transform back to [-1, 1]
    return 2 * half_angle - 1;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
