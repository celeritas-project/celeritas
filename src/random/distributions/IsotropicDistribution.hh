//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file IsotropicDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>
#include "base/Constants.hh"
#include "base/Array.hh"
#include "base/Types.hh"
#include "UniformRealDistribution.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sample points uniformly on the surface of a unit sphere.
 */
template<class RealType = ::celeritas::real_type>
class IsotropicDistribution
{
  public:
    //!@{
    //! Type aliases
    using real_type   = RealType;
    using result_type = Array<real_type, 3>;
    //!@}

  public:
    // Constructor
    inline CELER_FUNCTION IsotropicDistribution();

    // Sample a random unit vector
    template<class Generator>
    inline CELER_FUNCTION result_type operator()(Generator& rng);

  private:
    UniformRealDistribution<real_type> sample_costheta_;
    UniformRealDistribution<real_type> sample_phi_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with defaults.
 */
template<class RealType>
CELER_FUNCTION IsotropicDistribution<RealType>::IsotropicDistribution()
    : sample_costheta_(-1, 1), sample_phi_(0, 2 * constants::pi)
{
}

//---------------------------------------------------------------------------//
/*!
 * Sample an isotropic unit vector.
 */
template<class RealType>
template<class Generator>
CELER_FUNCTION auto IsotropicDistribution<RealType>::operator()(Generator& rng)
    -> result_type
{
    const real_type costheta = sample_costheta_(rng);
    const real_type phi      = sample_phi_(rng);
    const real_type sintheta = std::sqrt(1 - costheta * costheta);
    return {sintheta * std::cos(phi), sintheta * std::sin(phi), costheta};
}

//---------------------------------------------------------------------------//
} // namespace celeritas
