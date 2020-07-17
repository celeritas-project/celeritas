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

  public:
    // Construct from state
    __device__ inline RngEngine(const RngStatePointers& view,
                                const ThreadId&         id);

    // Sample a random number
    __device__ inline result_type operator()();

    // RNG state; use for implementation only!
    __device__ inline RngState* state() { return state_; }

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

//---------------------------------------------------------------------------//
