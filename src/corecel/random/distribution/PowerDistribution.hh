//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/distribution/PowerDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>
#include <type_traits>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"

#include "UniformRealDistribution.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sample a power distribution for powers not equal to -1.
 *
 * This distribution for a power \f$ p != -1 \f$ is defined on a positive range
 * \f$ [a, b) \f$ and has the normalized PDF:
 * \f[
   f(x; p, a, b) = x^{p}\frac{p + 1}{b^{p + 1} - a^{p + 1}}
   \quad \mathrm{for } a < x < b
   \f]
 * which integrated into a CDF and inverted gives a sample:
 * \f[
    x = \left[ (b^{p + 1} - a^{p + 1})\xi + a^{p + 1} \right]^{1/(p + 1)}
   \f]
 *
 * The quantity in brackets is a uniform sample on
 * \f$ [a^{p + 1}, b^{p + 1}) \f$
 *
 * See C13A (and C15A) from \citet{everett-montecarlo-1972,
 * https://doi.org/10.2172/4589395} , with \f$ x \gets u \f$,
 * \f$ p \gets m - 1 \f$.
 *
 * \note For \f$ p = -1 \f$ see \c ReciprocalDistribution ,
 * and in the degenerate case of \f$ p = 0 \f$ this is mathematically
 * equivalent to \c UniformRealDistribution .
 */
template<class RealType = ::celeritas::real_type>
class PowerDistribution
{
  public:
    //!@{
    //! \name Type aliases
    using real_type = RealType;
    using result_type = real_type;
    //!@}

  public:
    // Construct on an the interval [0, 1)
    explicit inline CELER_FUNCTION PowerDistribution(real_type p);

    // Construct on an arbitrary interval
    inline CELER_FUNCTION
    PowerDistribution(real_type p, real_type a, real_type b);

    // Sample a random number according to the distribution
    template<class Generator>
    inline CELER_FUNCTION result_type operator()(Generator& rng) const;

  private:
    UniformRealDistribution<real_type> sample_before_exp_;
    real_type exp_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct on the interval [0, 1).
 */
template<class RealType>
CELER_FUNCTION PowerDistribution<RealType>::PowerDistribution(real_type p)
    : sample_before_exp_{}, exp_{1 / (p + 1)}
{
    CELER_EXPECT(p != -1);
}

//---------------------------------------------------------------------------//
/*!
 * Construct on the interval [a, b).
 *
 * It is allowable for the two bounds to be out of order.
 */
template<class RealType>
CELER_FUNCTION PowerDistribution<RealType>::PowerDistribution(real_type p,
                                                              real_type a,
                                                              real_type b)
    : sample_before_exp_{fastpow(a, p + 1), fastpow(b, p + 1)}
    , exp_{1 / (p + 1)}
{
    CELER_EXPECT(p != -1);
    CELER_EXPECT(a >= 0);
    CELER_EXPECT(b >= 0);
}

//---------------------------------------------------------------------------//
/*!
 * Sample a random number according to the distribution.
 */
template<class RealType>
template<class Generator>
CELER_FUNCTION auto
PowerDistribution<RealType>::operator()(Generator& rng) const -> result_type
{
    real_type xi = sample_before_exp_(rng);
    return fastpow(xi, exp_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
