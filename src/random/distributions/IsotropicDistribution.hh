//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file IsotropicDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include "UniformRealDistribution.hh"
#include "base/Array.hh"
#include "base/Types.hh"

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
} // namespace celeritas

#include "IsotropicDistribution.i.hh"
