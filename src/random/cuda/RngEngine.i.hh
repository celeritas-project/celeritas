//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RngEngine.i.hh
//---------------------------------------------------------------------------//

#include "celeritas_config.h"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from state.
 */
CELER_FUNCTION
RngEngine::RngEngine(const RngStatePointers& view, const ThreadId& id)
    : state_(view.rng[id.get()])
{
    CELER_EXPECT(id < view.rng.size());
}

//---------------------------------------------------------------------------//
/*!
 * Initialize the RNG engine with a seed value.
 */
CELER_FUNCTION RngEngine& RngEngine::operator=(RngSeed s)
{
    curand_init(s.seed, 0, 0, &state_);
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Sample a random number.
 */
CELER_FUNCTION auto RngEngine::operator()() -> result_type
{
    return curand(&state_);
}

//---------------------------------------------------------------------------//
// Specializations for GenerateCanonical
//---------------------------------------------------------------------------//
/*!
 * Specialization for RngEngine (float).
 */
CELER_FUNCTION float
GenerateCanonical<RngEngine, float>::operator()(RngEngine& rng)
{
    return curand_uniform(&rng.state_);
}

//---------------------------------------------------------------------------//
/*!
 * Specialization for RngEngine (double).
 */
CELER_FUNCTION double
GenerateCanonical<RngEngine, double>::operator()(RngEngine& rng)
{
    return curand_uniform_double(&rng.state_);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
