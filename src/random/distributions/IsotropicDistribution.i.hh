//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file IsotropicDistribution.i.hh
//---------------------------------------------------------------------------//

#include <cmath>
#include "base/Constants.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with defaults.
 */
template<typename T>
CELER_FUNCTION IsotropicDistribution<T>::IsotropicDistribution()
    : sample_costheta_(-1, 1), sample_phi_(0, 2 * constants::pi)
{
}

//---------------------------------------------------------------------------//
/*!
 * Sample a random unit vector.
 */
template<class T>
template<class Generator>
CELER_FUNCTION auto IsotropicDistribution<T>::operator()(Generator& rng)
    -> result_type
{
    real_type costheta = sample_costheta_(rng);
    real_type phi      = sample_phi_(rng);
    real_type sintheta = std::sqrt(1 - costheta * costheta);
    return {sintheta * std::cos(phi), sintheta * std::sin(phi), costheta};
}

//---------------------------------------------------------------------------//
} // namespace celeritas
