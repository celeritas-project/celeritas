//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ExponentialDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sample from an exponential distribution: -log(xi) / lambda.
 *
 * This is simply an implementation of std::exponential_distribution that uses
 * the Celeritas canonical generator and is independent of library
 * implementation.
 *
 * Note (for performance-critical sections of code) that if this class is
 * constructed locally with the default value of lambda = 1.0, the
 * inversion and multiplication will be optimized out (and the code will be
 * exactly identical to `-std::log(rng.ran())`.
 */
template<class RealType = ::celeritas::real_type>
class ExponentialDistribution
{
  public:
    //!@{
    //! Type aliases
    using real_type   = RealType;
    using result_type = RealType;
    //!@}

  public:
    // Construct with defaults
    explicit inline CELER_FUNCTION
    ExponentialDistribution(real_type lambda = 1);

    // Sample using the given random number generator
    template<class Generator>
    inline CELER_FUNCTION result_type operator()(Generator& g);

  private:
    result_type neg_inv_lambda_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "ExponentialDistribution.i.hh"
