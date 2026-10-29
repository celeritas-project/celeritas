//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/interactor/RayleighInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/Assert.hh"
#include "corecel/random/engine/RanluxppRngEngine.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Advances the RNG engine if the engine is RanluxppRngEngine
 */
class RNGInteractor
{
  public:
    // Construct interactor from an optical track
    inline CELER_FUNCTION RNGInteractor() = default;

    // Sample an interaction with the given RNG
    inline CELER_FUNCTION void operator()(celeritas::RanluxppRngEngine& rng);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Sample a single optical Rayleigh interaction.
 */
CELER_FUNCTION void RNGInteractor::operator()(celeritas::RanluxppRngEngine& rng)
{
    // Force the RNG to create a new block of bits in preparation for taking a
    // bunch of samples
    rng.advance();
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
