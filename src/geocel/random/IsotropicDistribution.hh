//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/random/IsotropicDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Constants.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/random/distribution/UniformRealDistribution.hh"

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
    //! \name Type aliases
    using real_type = RealType;
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
    : sample_costheta_(-1, 1), sample_phi_(0, real_type(2 * constants::pi))
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
    real_type const costheta = sample_costheta_(rng);
    real_type const phi = sample_phi_(rng);
    return from_spherical(costheta, phi);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
