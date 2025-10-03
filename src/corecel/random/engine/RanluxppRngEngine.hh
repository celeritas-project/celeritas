//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/engine/RanluxppRngEngine.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstdint>
#include <string>

#include "corecel/Assert.hh"
#include "corecel/random/data/RanluxppRngData.hh"
#include "corecel/random/data/detail/RanluxppLCG.hh"
#include "corecel/random/data/detail/RanluxppMulMod.hh"
#include "corecel/sys/ThreadId.hh"

namespace celeritas
{

//---------------------------------------------------------------------------//
/*!
 * Implementation of the RanluxPP RNG Engine
 *
 * \tparam w  Position offset amount in bits
 */
template<int w>
class RanluxppRngEngineImpl
{
  public:
    //!{
    //! Type aliases
    using ParamsRef = NativeCRef<RanluxppRngParamsData>;
    using StateRef = NativeRef<RanluxppRngStateData>;
    using result_type = RanluxppUInt;
    //!}

  protected:
    // Return the next random bits, generate a new block if necessary
    CELER_FUNCTION inline RanluxppUInt nextRandomBits();

    //! Initialize and seed the state of the generator
    CELER_FUNCTION void setSeed(RanluxppUInt s)
    {
        state_->initialize(s, params_.kA_2048);
    }

    //! Skip `n` random numbers without generating them
    CELER_FUNCTION void skip(RanluxppUInt n)
    {
        CELER_ASSERT(n > 0);
        state_->skip(n, params_.kMaxPos, params_.kA_2048, w);
    }

  public:
    //! Lowest value potentially generated (check this)
    static CELER_CONSTEXPR_FUNCTION result_type min() { return 0u; }

    //! Highest value potentially generated (check this)
    static CELER_CONSTEXPR_FUNCTION result_type max()
    {
        return celeritas::numeric_limits<RanluxppUInt>::max();
    }

    // Construct from state and persistent data
    CELER_FUNCTION RanluxppRngEngineImpl(ParamsRef const& params,
                                         StateRef const& state,
                                         TrackSlotId tid)
        : params_(params)
    {
        CELER_EXPECT(tid < state.state.size());
        state_ = &state.state[tid];
    }

  private:
    /// DATA ///
    ParamsRef const& params_;
    RanluxppRngState* state_;
};

//---------------------------------------------------------------------------//

class RanluxppRngEngine final : public RanluxppRngEngineImpl<48>
{
    using Base = RanluxppRngEngineImpl<48>;

  public:
    //@{
    //! Public types
    using result_type = typename Base::result_type;
    using ParamsRef = typename Base::ParamsRef;
    using StateRef = typename Base::StateRef;
    //@}

  public:
    //! Instantiate with optional default seed
    CELER_FUNCTION RanluxppRngEngine(ParamsRef const& params,
                                     StateRef const& state,
                                     TrackSlotId tid)
        : Base(params, state, tid)
    {
        /* * */
    }

    //! Initialize state with the given seed
    inline CELER_FUNCTION RanluxppRngEngine& operator=(RanluxppUInt seed)
    {
        Base::setSeed(seed);
        return *this;
    }

    //! Generate a double-precision random number
    CELER_FUNCTION RanluxppUInt operator()() { return this->intRndm64(); }

    //! Advance the state \c count times
    inline CELER_FUNCTION void discard(RanluxppUInt count)
    {
        // Have to discard twice because 64-bit random numbers are composed of
        // *two* calls to nextRandomBits
        Base::skip(2 * count);
    }

  private:
    //! Generate a uniformly random 64-bit integer by concatenating two
    //! 32-bit words
    CELER_FUNCTION RanluxppUInt intRndm64()
    {
        // draw two 48-bit words, but take only their low 32 bits each
        RanluxppUInt lo = this->nextRandomBits() & 0xFFFFFFFFu;
        RanluxppUInt hi = this->nextRandomBits() & 0xFFFFFFFFu;
        return (lo << 32) | hi;
    }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Return the next random bits, generate a new block if necessary
 */
template<int w>
CELER_FUNCTION RanluxppUInt RanluxppRngEngineImpl<w>::nextRandomBits()
{
    if (state_->position + w > params_.kMaxPos)
    {
        state_->advance(params_.kA_2048);
    }

    int idx = state_->position / 64;
    int offset = state_->position % 64;
    int numBits = 64 - offset;

    RanluxppUInt bits = state_->state[idx] >> offset;
    if (numBits < w)
    {
        bits |= state_->state[idx + 1] << numBits;
    }
    bits &= ((RanluxppUInt(1) << w) - 1);

    state_->position += w;
    CELER_ASSERT(state_->position <= params_.kMaxPos);

    return bits;
}

//---------------------------------------------------------------------------//
}  // end namespace celeritas
