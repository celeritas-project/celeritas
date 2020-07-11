//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GenerateCanonical.hh
//---------------------------------------------------------------------------//
#ifndef random_GenerateCanonical_hh
#define random_GenerateCanonical_hh

#ifdef __NVCC__
#    include "RngEngine.cuh"
#endif
#include <random>
#include "base/Macros.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Generate random numbers in [0, 1).
 */
template<class Generator, class RealType = double>
class GenerateCanonical
{
  public:
    //@{
    //! Type aliases
    using real_type   = RealType;
    using result_type = real_type;
    //@}

  public:
    // Constructor
    explicit CELER_FUNCTION GenerateCanonical() {}

    // Sample a random number
    CELER_FUNCTION result_type operator()(Generator& rng);
};

#ifdef __NVCC__
//---------------------------------------------------------------------------//
/*!
 * Specialization for RngEngine, float
 */
template<>
class GenerateCanonical<RngEngine, float>
{
  public:
    //@{
    //! Type aliases
    using real_type   = float;
    using result_type = real_type;
    //@}

  public:
    // Constructor
    explicit CELER_FUNCTION GenerateCanonical() {}

    // Sample a random number
    inline __device__ result_type operator()(RngEngine& rng);
};

//---------------------------------------------------------------------------//
/*!
 * Specialization for RngEngine, double
 */
template<>
class GenerateCanonical<RngEngine, double>
{
  public:
    //@{
    //! Type aliases
    using real_type   = double;
    using result_type = real_type;
    //@}

  public:
    // Constructor
    explicit CELER_FUNCTION GenerateCanonical() {}

    // Sample a random number
    inline __device__ result_type operator()(RngEngine& rng);
};

#endif
//---------------------------------------------------------------------------//
} // namespace celeritas

#include "GenerateCanonical.i.hh"

#endif // random_GenerateCanonical_cuh
