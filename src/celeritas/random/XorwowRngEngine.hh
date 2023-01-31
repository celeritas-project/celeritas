//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/XorwowRngEngine.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "corecel/sys/ThreadId.hh"

#include "XorwowRngData.hh"
#include "detail/GenerateCanonical32.hh"
#include "distribution/GenerateCanonical.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Generate random data using the XORWOW algorithm.
 *
 * The XorwowRngEngine uses a C++11-like interface to generate random data. The
 * sampling of uniform floating point data is done with specializations to the
 * GenerateCanonical class.
 *
 * This class does not define an initializer because it is assumed that the
 * state has been fully randomized at initialization (see the \c resize
 * function for \c XorwowRngStateData.)
 *
 * See Marsaglia (2003) for the theory underlying the algorithm and the the
 * "example" \c xorwow that combines an \em xorshift output with a Weyl
 * sequence.
 *
 * https://www.jstatsoft.org/index.php/jss/article/view/v008i14/916
 */
class XorwowRngEngine
{
  public:
    //!@{
    //! \name Type aliases
    using result_type = unsigned int;
    using StateRef = NativeRef<XorwowRngStateData>;
    //!@}

  public:
    //! Lowest value potentially generated
    static CELER_CONSTEXPR_FUNCTION result_type min() { return 0u; }
    //! Highest value potentially generated
    static CELER_CONSTEXPR_FUNCTION result_type max() { return 0xffffffffu; }

    // Construct from state
    inline CELER_FUNCTION
    XorwowRngEngine(StateRef const& state, ThreadId const& id);

    // Generate a 32-bit pseudorandom number
    inline CELER_FUNCTION result_type operator()();

  private:
    XorwowState* state_;
};

//---------------------------------------------------------------------------//
/*!
 * Specialization of GenerateCanonical for XorwowRngEngine.
 */
template<class RealType>
class GenerateCanonical<XorwowRngEngine, RealType>
{
  public:
    //!@{
    //! \name Type aliases
    using real_type = RealType;
    using result_type = RealType;
    //!@}

  public:
    //! Sample a random number on [0, 1)
    CELER_FORCEINLINE_FUNCTION result_type operator()(XorwowRngEngine& rng)
    {
        return detail::GenerateCanonical32<RealType>()(rng);
    }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from state.
 */
CELER_FUNCTION
XorwowRngEngine::XorwowRngEngine(StateRef const& state, ThreadId const& id)
{
    CELER_EXPECT(id < state.state.size());
    state_ = &state.state[id];
}

//---------------------------------------------------------------------------//
/*!
 * Generate a 32-bit pseudorandom number using the 'xorwow' engine.
 */
CELER_FUNCTION auto XorwowRngEngine::operator()() -> result_type
{
    auto& s = state_->xorstate;
    auto const t = (s[0] ^ (s[0] >> 2u));

    s[0] = s[1];
    s[1] = s[2];
    s[2] = s[3];
    s[3] = s[4];
    s[4] = (s[4] ^ (s[4] << 4u)) ^ (t ^ (t << 1u));

    state_->weylstate += 362437u;
    return state_->weylstate + s[4];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
