//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RngEngine.i.cuh
//---------------------------------------------------------------------------//

#include <curand_kernel.h>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from state.
 */
__device__
RngEngine::RngEngine(const RngStatePointers& view, const ThreadId& id)
    : state_(view.rng[id.get()])
{
    REQUIRE(id < view.rng.size());
}

//---------------------------------------------------------------------------//
/*!
 * Initialize the RNG engine with a seed value.
 */
__device__ RngEngine& RngEngine::operator=(RngSeed s)
{
    curand_init(s.seed, 0, 0, &state_);
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Sample a random number
 */
__device__ auto RngEngine::operator()() -> result_type
{
    return curand(&state_);
}

//---------------------------------------------------------------------------//
// Specializations for GenerateCanonical
//---------------------------------------------------------------------------//
/*!
 * Specialization for RngEngine, float
 */
__device__ float GenerateCanonical<RngEngine, float>::operator()(RngEngine& rng)
{
    return curand_uniform(&rng.state_);
}

//---------------------------------------------------------------------------//
/*!
 * Specialization for RngEngine, double
 */
__device__ double
GenerateCanonical<RngEngine, double>::operator()(RngEngine& rng)
{
    return curand_uniform_double(&rng.state_);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
