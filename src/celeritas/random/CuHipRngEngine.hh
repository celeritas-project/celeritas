//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/CuHipRngEngine.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/sys/ThreadId.hh"

#include "CuHipRngData.hh"
#include "distribution/GenerateCanonical.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Generate random data on device and host.
 *
 * The CuHipRngEngine uses a C++11-like interface to generate random data. The
 * sampling of uniform floating point data is done with specializations to the
 * GenerateCanonical class.
 */
class CuHipRngEngine
{
  public:
    //!@{
    //! \name Type aliases
    using result_type = unsigned int;
    using Initializer_t = CuHipRngInitializer;
    using ParamsRef = NativeCRef<CuHipRngParamsData>;
    using StateRef = NativeRef<CuHipRngStateData>;
    //!@}

  public:
    // Construct from state
    inline CELER_FUNCTION CuHipRngEngine(ParamsRef const& params,
                                         StateRef const& state,
                                         TrackSlotId tid);

    // Initialize state from seed
    inline CELER_FUNCTION CuHipRngEngine& operator=(Initializer_t const& s);

    // Sample a random number
    inline CELER_FUNCTION result_type operator()();

  private:
    ParamsRef const& params_;
    CuHipRngThreadState* state_;

    template<class Generator, class RealType>
    friend class GenerateCanonical;
};

//---------------------------------------------------------------------------//
/*!
 * Specialization of GenerateCanonical for CuHipRngEngine, float
 */
template<>
class GenerateCanonical<CuHipRngEngine, float>
{
  public:
    //!@{
    //! \name Type aliases
    using real_type = float;
    using result_type = real_type;
    //!@}

  public:
    // Sample a random number
    inline CELER_FUNCTION result_type operator()(CuHipRngEngine& rng);
};

//---------------------------------------------------------------------------//
/*!
 * Specialization for CuHipRngEngine, double
 */
template<>
class GenerateCanonical<CuHipRngEngine, double>
{
  public:
    //!@{
    //! \name Type aliases
    using real_type = double;
    using result_type = real_type;
    //!@}

  public:
    // Sample a random number
    inline CELER_FUNCTION result_type operator()(CuHipRngEngine& rng);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from state.
 */
CELER_FUNCTION
CuHipRngEngine::CuHipRngEngine(ParamsRef const& params,
                               StateRef const& state,
                               TrackSlotId tid)
    : params_(params)
{
    CELER_EXPECT(tid < state.rng.size());
    state_ = &state.rng[tid];
}

//---------------------------------------------------------------------------//
/*!
 * Initialize the RNG engine with a seed value.
 */
CELER_FUNCTION CuHipRngEngine& CuHipRngEngine::operator=(Initializer_t const& s)
{
    CELER_RNG_PREFIX(rand_init)(s.seed, s.subsequence, s.offset, state_);
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Sample a random number.
 */
CELER_FUNCTION auto CuHipRngEngine::operator()() -> result_type
{
    return CELER_RNG_PREFIX(rand)(state_);
}

//---------------------------------------------------------------------------//
/*!
 * Specialization for CuHipRngEngine (float).
 */
CELER_FUNCTION float
GenerateCanonical<CuHipRngEngine, float>::operator()(CuHipRngEngine& rng)
{
    return CELER_RNG_PREFIX(rand_uniform)(rng.state_);
}

//---------------------------------------------------------------------------//
/*!
 * Specialization for CuHipRngEngine (double).
 */
CELER_FUNCTION double
GenerateCanonical<CuHipRngEngine, double>::operator()(CuHipRngEngine& rng)
{
    return CELER_RNG_PREFIX(rand_uniform_double)(rng.state_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
