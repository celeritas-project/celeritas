//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RadialDistribution.hh
//---------------------------------------------------------------------------//
#ifndef random_RadialDistribution_hh
#define random_RadialDistribution_hh

#include "base/Macros.hh"

#ifndef __NVCC__
#    include <random>
using UniformDistribution_t = std::uniform_real_distribution<double>;
#else
struct NotUniformRealDistribution
{
    double vmin, vmax;
    template<class G>
    CELER_INLINE_FUNCTION double operator()(G&)
    {
        return 1.0;
    }
};
using UniformDistribution_t = NotUniformRealDistribution;
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sample from a uniform radial distribution.
 */
template<class RealType = double>
class RadialDistribution
{
  public:
    //@{
    //! Type aliases
    using real_type   = RealType;
    using result_type = real_type;
    //@}

  public:
    // Constructor
    explicit CELER_INLINE_FUNCTION RadialDistribution(real_type radius);

    // Sample a random number according to the distribution
    template<class Generator>
    CELER_INLINE_FUNCTION result_type operator()(Generator& rng);

    // >>> ACCESSORS

    //! Get the sampling radius
    CELER_INLINE_FUNCTION real_type radius() const { return radius_; }

  private:
    RealType              radius_;
    UniformDistribution_t sample_uniform_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "RadialDistribution.i.hh"

#endif // random_RadialDistribution_hh
