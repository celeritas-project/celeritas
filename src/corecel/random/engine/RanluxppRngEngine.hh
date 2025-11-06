//------------------------------- -*- C++ -*- -------------------------------//
// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0
//---------------------------------------------------------------------------//
/*!
 * \file corecel/random/engine/RanluxppRngEngine.hh
 *
 * Original source:
 * https://github.com/apt-sim/AdePT/blob/master/include/AdePT/copcore/Ranluxpp.h
 */
//---------------------------------------------------------------------------//
#pragma once

#include <cstdint>
#include <string>

#include "corecel/Assert.hh"
#include "corecel/random/data/RanluxppRngData.hh"
#include "corecel/random/data/detail/RanluxppImpl.hh"
#include "corecel/sys/ThreadId.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Implements the Ranluxpp random number generator engine.
 *
 * A paper detailing this random number generator can be found here:
 * \citet{hahnfeld-ranlux-2021, https://doi.org/10.1051/epjconf/202125103008}.
 */
class RanluxppRngEngine
{
  public:
    //@{
    //! Public types.
    using ParamsRef = NativeCRef<RanluxppRngParamsData>;
    using StateRef = NativeRef<RanluxppRngStateData>;
    using result_type = RanluxppUInt;
    //@}

  public:
    //! Instantiate with optional default seed.
    CELER_FUNCTION RanluxppRngEngine(ParamsRef const& params,
                                     StateRef const& state,
                                     TrackSlotId tid)
        : params_(params)
    {
        CELER_EXPECT(tid < state.state.size());
        state_ = &state.state[tid];
    }

    //! Lowest value potentially generated.
    static CELER_CONSTEXPR_FUNCTION result_type min() { return 0u; }

    //! Highest value potentially generated.
    static CELER_CONSTEXPR_FUNCTION result_type max()
    {
        return celeritas::numeric_limits<RanluxppUInt>::max();
    }

    //! Initialize state with the given seed.
    inline CELER_FUNCTION RanluxppRngEngine& operator=(RanluxppInitializer init)
    {
        // Skip forward (params_.seed + init.thread_local_id) states
        RanluxppArray9 new_a_seed = celeritas::detail::compute_power_modulus(
            params_.seed_state, params_.seed + init.thread_local_id);
        RanluxppArray9 lcg = {1, 0, 0, 0, 0, 0, 0, 0, 0};
        lcg = celeritas::detail::compute_mod_multiply(new_a_seed, lcg);

        // Convert to Ranluxpp number and save state
        state_->value = celeritas::detail::to_ranlux(lcg);
        state_->position = 0;
        return *this;
    }

    //! Generate a double-precision random number.
    CELER_FUNCTION RanluxppUInt operator()() { return this->intRndm64(); }

    //! Advance the state \c count times.
    inline CELER_FUNCTION void discard(RanluxppUInt count)
    {
        // Have to discard twice because 64-bit random numbers are composed of
        // *two* calls to nextRandomBits
        this->skip(2 * count);
    }

  private:
    /// IMPLEMENTATION ///

    // Skip 'n' random numbers without generating them
    inline CELER_FUNCTION void skip(RanluxppUInt n);

    // Return the next random bits, generate a new block if necessary
    inline CELER_FUNCTION RanluxppUInt nextRandomBits();

    //! Generate a uniformly random 64-bit integer by concatenating two
    //! 32-bit words.
    CELER_FUNCTION RanluxppUInt intRndm64()
    {
        // draw two 48-bit words, but take only their low 32 bits each
        RanluxppUInt lo = this->nextRandomBits() & 0xFFFFFFFFu;
        RanluxppUInt hi = this->nextRandomBits() & 0xFFFFFFFFu;
        return (lo << 32) | hi;
    }

    //! Produce the next block of random bits.
    CELER_FUNCTION void advance()
    {
        RanluxppArray9 lcg = celeritas::detail::to_lcg(state_->value);
        lcg = celeritas::detail::compute_mod_multiply(params_.state_2048, lcg);
        state_->value = celeritas::detail::to_ranlux(lcg);
        state_->position = 0;
    }

    /// DATA ///
    static constexpr int offset_ = 48;
    ParamsRef const& params_;
    RanluxppRngState* state_;
};

//---------------------------------------------------------------------------//
// INLINE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Skip `n` random numbers without generating them.
 */
CELER_FUNCTION void RanluxppRngEngine::skip(RanluxppUInt n)
{
    CELER_ASSERT(n > 0);
    CELER_ASSERT(params_.max_position > 0);

    int left = (params_.max_position - state_->position) / offset_;
    CELER_ASSERT(left >= 0);
    if (n < static_cast<RanluxppUInt>(left))
    {
        // Just skip the next few entries in the currently
        // available bits.
        state_->position += n * offset_;
        CELER_ASSERT(state_->position <= params_.max_position);
        return;
    }

    n -= left;
    // Need to advance and possibly skip over blocks.
    int nPerState = params_.max_position / offset_;
    int skip = n / nPerState;

    RanluxppArray9 a_skip = celeritas::detail::compute_power_modulus(
        params_.state_2048, skip + 1);

    RanluxppArray9 lcg = celeritas::detail::to_lcg(state_->value);
    lcg = celeritas::detail::compute_mod_multiply(a_skip, lcg);
    state_->value = celeritas::detail::to_ranlux(lcg);

    // Potentially skip numbers in the freshly generated block.
    int remaining = n - skip * nPerState;
    CELER_ASSERT(remaining >= 0);
    state_->position = remaining * offset_;
    CELER_ASSERT(state_->position <= params_.max_position);
}

//---------------------------------------------------------------------------//
/*!
 * Return the next random bits, generate a new block if necessary.
 */
CELER_FUNCTION RanluxppUInt RanluxppRngEngine::nextRandomBits()
{
    if (state_->position + offset_ > params_.max_position)
    {
        this->advance();
    }

    int idx = state_->position / 64;
    int offset = state_->position % 64;
    int numBits = 64 - offset;

    RanluxppUInt bits = state_->value.number[idx] >> offset;
    if (numBits < offset_)
    {
        bits |= state_->value.number[idx + 1] << numBits;
    }
    bits &= ((RanluxppUInt(1) << offset_) - 1);

    state_->position += offset_;
    CELER_ASSERT(state_->position <= params_.max_position);

    return bits;
}

//---------------------------------------------------------------------------//
}  // end namespace celeritas
