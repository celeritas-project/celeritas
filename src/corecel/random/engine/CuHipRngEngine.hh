//------------------------------ -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/engine/CuHipRngEngine.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/random/data/CuHipRngData.hh"
#include "corecel/random/distribution/GenerateCanonical.hh"
#include "corecel/sys/ThreadId.hh"

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
    inline CELER_FUNCTION CuHipRngEngine& operator=(Initializer_t const&);

    // Sample a random number
    inline CELER_FUNCTION result_type operator()();

  private:
    CuHipRngThreadState* state_;

    template<class Generator, class RealType>
    friend struct GenerateCanonical;
};

//---------------------------------------------------------------------------//
/*!
 * Specialization of GenerateCanonical for CuHipRngEngine, float
 */
template<>
struct GenerateCanonical<CuHipRngEngine, float>
{
    //!@{
    //! \name Type aliases
    using real_type = float;
    using result_type = real_type;
    //!@}

    //! Instead of using builtin canonical, we call CUDA/HIP
    static constexpr auto policy = GenerateCanonicalPolicy::custom;

    // Sample a random number
    inline CELER_FUNCTION result_type operator()(CuHipRngEngine& rng);
};

//---------------------------------------------------------------------------//
/*!
 * Specialization for CuHipRngEngine, double
 */
template<>
struct GenerateCanonical<CuHipRngEngine, double>
{
    //!@{
    //! \name Type aliases
    using real_type = double;
    using result_type = real_type;
    //!@}

    //! Instead of using builtin canonical, we call CUDA/HIP
    static constexpr auto policy = GenerateCanonicalPolicy::custom;

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
CuHipRngEngine::CuHipRngEngine(ParamsRef const&,
                               StateRef const& state,
                               TrackSlotId tid)
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
