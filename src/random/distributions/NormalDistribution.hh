//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file NormalDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"

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
    //! Type aliases
    using real_type   = RealType;
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
    const real_type mean_;
    const real_type stddev_;
    real_type       spare_;
    bool            has_spare_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "NormalDistribution.i.hh"
