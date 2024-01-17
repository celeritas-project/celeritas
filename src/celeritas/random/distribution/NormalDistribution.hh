//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/distribution/NormalDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/Constants.hh"

#include "GenerateCanonical.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sample from a normal distribution.
 *
 * This uses the Box-Muller transform to generate pairs of independent,
 * normally distributed random numbers, returning them one at a time. Two
 * random numbers uniformly distributed on [0, 1] are mapped to two
 * independent, standard, normally distributed samples using the relations:
 * \f[
  x_1 = \sqrt{-2 \ln \xi_1} \cos(2 \pi \xi_2)
  x_2 = \sqrt{-2 \ln \xi_1} \sin(2 \pi \xi_2)
   \f]
 */
template<class RealType = ::celeritas::real_type>
class NormalDistribution
{
  public:
    //!@{
    //! \name Type aliases
    using real_type = RealType;
    using result_type = real_type;
    //!@}

  public:
    // Construct with mean and standard deviation
    explicit inline CELER_FUNCTION
    NormalDistribution(real_type mean = 0, real_type stddev = 1);

    // Sample a random number according to the distribution
    template<class Generator>
    inline CELER_FUNCTION result_type operator()(Generator& rng);

  private:
    real_type const mean_;
    real_type const stddev_;
    real_type spare_{};
    bool has_spare_{false};
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with mean and standard deviation.
 */
template<class RealType>
CELER_FUNCTION
NormalDistribution<RealType>::NormalDistribution(real_type mean,
                                                 real_type stddev)
    : mean_(mean), stddev_(stddev)
{
    CELER_EXPECT(stddev > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Sample a random number according to the distribution.
 */
template<class RealType>
template<class Generator>
CELER_FUNCTION auto NormalDistribution<RealType>::operator()(Generator& rng)
    -> result_type
{
    if (has_spare_)
    {
        has_spare_ = false;
        return std::fma(spare_, stddev_, mean_);
    }

    constexpr auto twopi = static_cast<RealType>(2 * m_pi);
    real_type theta = twopi * generate_canonical<RealType>(rng);
    real_type r = std::sqrt(-2 * std::log(generate_canonical<RealType>(rng)));
    spare_ = r * std::cos(theta);
    has_spare_ = true;
    return std::fma(r * std::sin(theta), stddev_, mean_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
