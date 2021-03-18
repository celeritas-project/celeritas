//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RadialDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sample from a uniform radial distribution.
 */
template<class RealType = ::celeritas::real_type>
class RadialDistribution
{
  public:
    //!@{
    //! Type aliases
    using real_type   = RealType;
    using result_type = real_type;
    //!@}

  public:
    // Constructor
    explicit inline CELER_FUNCTION RadialDistribution(real_type radius);

    // Sample a random number according to the distribution
    template<class Generator>
    inline CELER_FUNCTION result_type operator()(Generator& rng);

    //// ACCESSORS ////

    //! Get the sampling radius
    inline CELER_FUNCTION real_type radius() const { return radius_; }

  private:
    RealType radius_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "RadialDistribution.i.hh"
