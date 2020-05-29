//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RngEngine.cuh
//---------------------------------------------------------------------------//
#ifndef random_RngEngine_cuh
#define random_RngEngine_cuh

#include <curand_kernel.h>
#include <cstddef>

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
    using size_type   = std::size_t;
    using seed_type   = unsigned long long;
    using RngState    = curandState_t;
    //@}

  public:
    // Construct from state
    __device__ inline RngEngine(RngState* state);

    // Sample a random number
    __device__ inline result_type operator()();

    // Sample a random number uniformly distributed on (0.0, 1.0]
    template<class ResultType = double>
    __device__ inline ResultType sample_uniform();

    // Initialize state from seed
    __device__ void initialize_state(seed_type seed)
    {
        curand_init(seed, 0, 0, state_);
    }

  private:
    RngState* state_ = nullptr;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "RngEngine.i.cuh"

#endif // random_RngEngine_cuh
