//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file UniformRealDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sample from a uniform distribution.
 *
 * This distribution is defined between two arbitrary real numbers \em a and
 * \em b , and has a flat PDF between the two values. It *is* allowable for the
 * two numbers to have reversed order.
 */
template<class RealType = ::celeritas::real_type>
class UniformRealDistribution
{
  public:
    //!@{
    //! Type aliases
    using real_type   = RealType;
    using result_type = real_type;
    //!@}

  public:
    // Construct on [0, 1)
    inline CELER_FUNCTION UniformRealDistribution();

    // Construct on an arbitrary interval
    explicit inline CELER_FUNCTION
    UniformRealDistribution(real_type a, real_type b = 1);

    // Sample a random number according to the distribution
    template<class Generator>
    inline CELER_FUNCTION result_type operator()(Generator& rng);

    //// ACCESSORS ////

    //! Get the lower bound of the distribution
    CELER_FUNCTION real_type a() const { return a_; }

    //! Get the upper bound of the distribution
    CELER_FUNCTION real_type b() const { return delta_ + a_; }

  private:
    RealType a_;
    RealType delta_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "UniformRealDistribution.i.hh"
