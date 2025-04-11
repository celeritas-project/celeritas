//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/distribution/BernoulliDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"

#include "GenerateCanonical.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Select one of two options with a given probability.
 *
 * The constructor argument is the chance of returning `true`, with an optional
 * second argument for normalizing with a fraction of `false` values.
 * \code
    BernoulliDistribution snake_eyes(1, 35);
    if (snake_eyes(rng))
    {
        // ...
    }
    BernoulliDistribution also_snake_eyes(1.0 / 36.0);
   \endcode
 */
class BernoulliDistribution
{
  public:
    //!@{
    //! \name Type aliases
    using result_type = bool;
    //!@}

  public:
    // Construct with the probability of returning true
    explicit inline CELER_FUNCTION BernoulliDistribution(real_type p_true);

    // Construct with the UNnormalized probability of returning true or false
    inline CELER_FUNCTION
    BernoulliDistribution(real_type scaled_true, real_type scaled_false);

    // Sample true or false based on the probability
    template<class Generator>
    inline CELER_FUNCTION result_type operator()(Generator& rng);

    //! Probability of returning `true` from operator()
    real_type p() const { return p_true_; }

  private:
    real_type p_true_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with the probability of returning true.
 */
CELER_FUNCTION BernoulliDistribution::BernoulliDistribution(real_type p_true)
    : p_true_(p_true)
{
    CELER_EXPECT(p_true >= 0 && p_true <= 1);
}

//---------------------------------------------------------------------------//
/*!
 * Construct with the UNnormalized probability of returning true or false
 */
CELER_FUNCTION
BernoulliDistribution::BernoulliDistribution(real_type scaled_true,
                                             real_type scaled_false)
    : p_true_(scaled_true / (scaled_true + scaled_false))
{
    CELER_EXPECT(scaled_true > 0 || scaled_false > 0);
    CELER_EXPECT(scaled_true >= 0 && scaled_false >= 0);
}

//---------------------------------------------------------------------------//
/*!
 * Construct with the probability of returning true.
 */
template<class Generator>
CELER_FUNCTION auto
BernoulliDistribution::operator()(Generator& rng) -> result_type
{
    return generate_canonical<real_type>(rng) < p_true_;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
