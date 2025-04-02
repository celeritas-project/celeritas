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

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sample the large-angle MSC scattering cosine.
 *
 * The Goudsmit and Saunderson moments for the expected angular deflection \f$
\theta \f$ over a true path length \f$ t \f$, are:
 * \f[
\langle \mu \rangle = \exp\left[-\frac{t}{\lambda_1}\right]
\f]
and
\f[
 \langle \cos^2 \theta \rangle =
\frac{1}{3}\left(1 + 2 \left[-\frac{t}{\lambda_2}\right]\right)
 \f]
 * where \f$ \lambda_l \f$ are transport mean free paths from the elastic cross
section scattering angular moments.

* Given the number of mean free paths \f[
\tau \equiv \frac{t}{\lambda_1}
\f]
* and from
\citet{kawrakow-condensedhistory-1998,https://doi.org/10.1016/S0168-583X(98)00274-2}
that for  kinetic energies between a few keV and infinity, \f[ 2 <
\frac{\lambda_2}{\lambda_1} < \infty
\f]
* this calculates the mean scattering angle and approximates the second moment
of the scattering cosine using \f$ \lambda_2 \approx 2.5 \lambda_1 \f$.
*
* Using these moments, \citet{urban-msc-2006,
https://cds.cern.ch/record/1004190/} proposes
 */
class UrbanLargeAngleDistribution
{
  public:
    // Construct with mean free path tau
    explicit inline CELER_FUNCTION UrbanLargeAngleDistribution(real_type tau);

    // Sample cos(theta)
    template<class Engine>
    inline CELER_FUNCTION real_type operator()(Engine& rng) const;

  private:
    real_type a_{};
    real_type pow_prob_{};
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
    // NOTE: tau_big~8 -> ~0.0003 < xmean < 1
    real_type musqmean = (1 + 2 * std::exp(real_type(-2.5) * tau)) / 3;

    a_ = (2 * mumean + 9 * musqmean - 3) / (2 * mumean - 3 * musqmean + 1);
    pow_prob_ = (a_ + 2) * mumean / a_;
}

//---------------------------------------------------------------------------//
/*!
 * Sample from two parameters of the model function.
 */
template<class Engine>
CELER_FUNCTION real_type UrbanLargeAngleDistribution::operator()(Engine& rng) const
{
    real_type result{};
    BernoulliDistribution sample_pow{pow_prob_};
    do
    {
        real_type rdm = generate_canonical(rng);
        result = 2 * (sample_pow(rng) ? fastpow(rdm, 1 / (a_ + 1)) : rdm) - 1;
    } while (std::fabs(result) > 1);

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
