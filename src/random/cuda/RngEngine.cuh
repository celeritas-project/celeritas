//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RngEngine.cuh
//---------------------------------------------------------------------------//
#pragma once

#include <curand_kernel.h>
#include "RngStatePointers.cuh"
#include "random/distributions/GenerateCanonical.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sample random numbers on device.
 */
class RngEngine
{
  public:
    //@{
    //! Type aliases
    using result_type = unsigned int;
    //@}

    struct RngSeed
    {
        unsigned long long seed;
    };

  public:
    // Construct from state
    __device__ inline RngEngine(const RngStatePointers& view,
                                const ThreadId&         id);

    // Initialize state from seed
    __device__ RngEngine& operator=(RngSeed s)
    {
        curand_init(s.seed, 0, 0, state_);
        return *this;
    }

    // Sample a random number
    __device__ inline result_type operator()();

  private:
    RngState* state_ = nullptr;

    template<class Generator, class RealType>
    friend class GenerateCanonical;
};

//---------------------------------------------------------------------------//
/*!
 * Specialization of GenerateCanonical for RngEngine, float
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
    // Sample a random number
    inline __device__ result_type operator()(RngEngine& rng);
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "RngEngine.i.cuh"

//---------------------------------------------------------------------------//
