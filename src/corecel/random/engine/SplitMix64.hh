//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/engine/SplitMix64.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstdint>

#include "corecel/Macros.hh"

namespace celeritas
{

//---------------------------------------------------------------------------//
/*!
 * \brief RNG for initializing the state for other RNGs
 *
 * See https://prng.di.unimi.it for a details of the SplitMix64 engine
 */
class SplitMix64
{
  public:
    // Construct the SplitMix64 class with the given seed
    inline CELER_FUNCTION explicit SplitMix64(std::uint64_t seed);

    // Produce a random number
    inline CELER_FUNCTION std::uint64_t operator()();

  private:
    // SplitMix64 State
    std::uint64_t state_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct the SplitMix64 engine with the given seed.
 */
CELER_FUNCTION SplitMix64::SplitMix64(std::uint64_t seed) : state_(seed)
{
    /* * */
}

//---------------------------------------------------------------------------//
/*!
 * Generate a 64-bit pseudorandom number using the SplitMix64 engine.
 *
 * See https://prng.di.unimi.it for a description of the method.
 */
CELER_FUNCTION std::uint64_t SplitMix64::operator()()
{
    std::uint64_t z = (state_ += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
