//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ReciprocalDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Reciprocal or log-uniform distribution.
 *
 * This distribution is defined on a positive range \f$ [a, b) \f$ and has the
 * normalized PDF:
 * \f[
   f(x; a, b) = \frac{1}{x (\ln b - \ln a)} \quad \mathrm{for} a \le x < b
   \f]
 * which integrated into a CDF and inverted gives a sample:
 * \f[
  x = a \left( \frac{b}{a} \right)^{\xi}
    = a \exp\!\left(\xi \log \frac{b}{a} \right)
   \f]
 */
template<class RealType = ::celeritas::real_type>
class ReciprocalDistribution
{
  public:
    //!@{
    //! Type aliases
    using real_type   = RealType;
    using result_type = real_type;
    //!@}

  public:
    // Construct on an the interval [a, 1]
    explicit inline CELER_FUNCTION ReciprocalDistribution(real_type a);

    // Construct on an arbitrary interval
    inline CELER_FUNCTION ReciprocalDistribution(real_type a, real_type b);

    // Sample a random number according to the distribution
    template<class Generator>
    inline CELER_FUNCTION result_type operator()(Generator& rng);

  private:
    RealType a_;
    RealType logratio_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "ReciprocalDistribution.i.hh"
