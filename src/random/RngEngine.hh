//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RngEngine.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Assert.hh"
#include "base/OpaqueId.hh"
#include "random/distributions/GenerateCanonical.hh"

#include "RngData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Generate random data on device and host.
 *
 * The RngEngine uses a C++11-like interface to generate random data. The
 * sampling of uniform floating point data is done with specializations to the
 * GenerateCanonical class.
 */
class RngEngine
{
  public:
    //!@{
    //! Type aliases
    using result_type   = unsigned int;
    using Initializer_t = RngInitializer<MemSpace::native>;
    using StateRef      = RngStateData<Ownership::reference, MemSpace::native>;
    //!@}

  public:
    // Construct from state
    inline CELER_FUNCTION RngEngine(const StateRef& state, const ThreadId& id);

    // Initialize state from seed
    inline CELER_FUNCTION RngEngine& operator=(const Initializer_t& s);

    // Sample a random number
    inline CELER_FUNCTION result_type operator()();

  private:
    RngThreadState* state_;

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
    //!@{
    //! Type aliases
    using real_type   = float;
    using result_type = real_type;
    //!@}

  public:
    // Sample a random number
    inline CELER_FUNCTION result_type operator()(RngEngine& rng);
};

//---------------------------------------------------------------------------//
/*!
 * Specialization for RngEngine, double
 */
template<>
class GenerateCanonical<RngEngine, double>
{
  public:
    //!@{
    //! Type aliases
    using real_type   = double;
    using result_type = real_type;
    //!@}

  public:
    // Sample a random number
    inline CELER_FUNCTION result_type operator()(RngEngine& rng);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from state.
 */
CELER_FUNCTION
RngEngine::RngEngine(const StateRef& state, const ThreadId& id)
{
    CELER_EXPECT(id < state.rng.size());
    state_ = &state.rng[id];
}

//---------------------------------------------------------------------------//
/*!
 * Initialize the RNG engine with a seed value.
 */
CELER_FUNCTION RngEngine& RngEngine::operator=(const Initializer_t& s)
{
    CELER_RNG_PREFIX(rand_init)(s.seed, 0, 0, state_);
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Sample a random number.
 */
CELER_FUNCTION auto RngEngine::operator()() -> result_type
{
    return CELER_RNG_PREFIX(rand)(state_);
}

//---------------------------------------------------------------------------//
/*!
 * Specialization for RngEngine (float).
 */
CELER_FUNCTION float
GenerateCanonical<RngEngine, float>::operator()(RngEngine& rng)
{
    return CELER_RNG_PREFIX(rand_uniform)(rng.state_);
}

//---------------------------------------------------------------------------//
/*!
 * Specialization for RngEngine (double).
 */
CELER_FUNCTION double
GenerateCanonical<RngEngine, double>::operator()(RngEngine& rng)
{
    return CELER_RNG_PREFIX(rand_uniform_double)(rng.state_);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
