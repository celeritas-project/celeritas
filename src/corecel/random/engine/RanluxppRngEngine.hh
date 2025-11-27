//------------------------------- -*- C++ -*- -------------------------------//
// SPDX-FileCopyrightText: 2020 CERN
// SPDX-FileCopyrightText: 2025 Celeritas contributors
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
/*!
 * \file corecel/random/engine/RanluxppRngEngine.hh
 *
 * Original source:
 * https://github.com/apt-sim/AdePT/blob/master/include/AdePT/copcore/Ranluxpp.h
 */
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/random/data/RanluxppRngData.hh"
#include "corecel/random/engine/detail/RanluxppImpl.hh"
#include "corecel/sys/ThreadId.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Implements the RANLUX++ random number generator engine with modifications.
 *
 * This implementation of RANLUX++
 * \citep{sibidanov-revisionsubtractwithborrow-2017,
 * https://doi.org/10.1016/j.cpc.2017.09.005} is based directly on the
 * implementation of
 * \citet{hahnfeld-ranlux-2021, https://doi.org/10.1051/epjconf/202125103008}.
 * The RANLUX++ algorithm
 * is an optimization of the RANLUX generator \citep{james-ranluxfortran-1994,
 * https://doi.org/10.1016/0010-4655(94)90233-X}, based on work by Luscher's
 * modification \citep{luscher-portablehighquality-1994,
 * http://arxiv.org/abs/hep-lat/9309020} of Marsaglia and Zaman's RCARRY
 * \citep{james-reviewpseudorandom-1990,
 * https://doi.org/10.1016/0010-4655(90)90032-V}. As discussed in the RANLUX
 * theory paper, the algorithm is essentially a linear congruential generator
 * (LCG) with a huge state.
 *
 * The underlying RCARRY algorithm used an array of 24 24-bit integer words,
 * which with today's large integer sizes can be written as 9 64-bit integers.
 * A given state is used to extract 12 samples, and the lower 32
 * bits of each is used as entropy. Two of those are concatenated to form a
 * 64-bit sample.
 *
 * \todo The decision to discard the high bits rather than the low bits from
 * each word is likely undesirable at least for the last number in the state,
 * since it is known that LCG integers have highly correlated sequential LSBs.
 * Additionally, the class should be adapted to provide 32-bit samples, which
 * the GenerateCanonical32 will map to 64-bit reals as needed.
 */
class RanluxppRngEngine
{
  public:
    //@{
    //! Public types.
    using result_type = RanluxppUInt;
    using Initializer_t = RanluxppInitializer;
    using ParamsRef = NativeCRef<RanluxppRngParamsData>;
    using StateRef = NativeRef<RanluxppRngStateData>;
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

    // Initialize state with the given seed.
    inline CELER_FUNCTION RanluxppRngEngine&
    operator=(Initializer_t const& init);

    // Generate a double-precision random number
    inline CELER_FUNCTION RanluxppUInt operator()();

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
    inline CELER_FUNCTION RanluxppUInt next_random_bits();

    // Produce the next block of random bits.
    inline CELER_FUNCTION void advance();

    /// DATA ///
    static constexpr int offset_ = 48;
    ParamsRef const& params_;
    RanluxppRngState* state_;
};

//---------------------------------------------------------------------------//
// INLINE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Initialize state for the given seed and subsequence
 */
inline CELER_FUNCTION RanluxppRngEngine&
RanluxppRngEngine::operator=(Initializer_t const& init)
{
    // Skip forward (2^96) * (init.seed + init.subsequence) states
    RanluxppArray9 new_a_seed = celeritas::detail::compute_power_modulus(
        params_.advance_sequence, init.seed + init.subsequence);

    // Convert to Ranluxpp number and save state
    state_->value = celeritas::detail::to_ranlux(new_a_seed);
    state_->position = 0;

    // Skip forward another init.offset samples
    if (init.offset > 0)
    {
        this->discard(init.offset);
    }
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Generate a double-precision random number
 */
CELER_FUNCTION RanluxppUInt RanluxppRngEngine::operator()()
{
    // draw two 48-bit words, but take only their low 32 bits each
    RanluxppUInt lo = this->next_random_bits() & 0xFFFFFFFFu;
    RanluxppUInt hi = this->next_random_bits() & 0xFFFFFFFFu;
    return (lo << 32) | hi;
}

//---------------------------------------------------------------------------//
/*!
 * Skip `n` random numbers without generating them.
 */
CELER_FUNCTION void RanluxppRngEngine::skip(RanluxppUInt n)
{
    CELER_EXPECT(n > 0);
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
    // Need to advance and possibly skip multiple blocks (each block is 576
    // random bits, or 12 48-bit samples)
    constexpr int n_per_state = ParamsRef::max_position / offset_;
    int skip = n / n_per_state;

    RanluxppArray9 a_skip = celeritas::detail::compute_power_modulus(
        params_.advance_state, skip + 1);

    RanluxppArray9 lcg = celeritas::detail::to_lcg(state_->value);
    lcg = celeritas::detail::compute_mod_multiply(a_skip, lcg);
    state_->value = celeritas::detail::to_ranlux(lcg);

    // Potentially skip numbers in the freshly generated block.
    int remaining = n - skip * n_per_state;
    CELER_ASSERT(remaining >= 0);
    state_->position = remaining * offset_;
    CELER_ENSURE(state_->position <= params_.max_position);
}

//---------------------------------------------------------------------------//
/*!
 * Return the next random bits, generate a new block if necessary.
 */
CELER_FUNCTION RanluxppUInt RanluxppRngEngine::next_random_bits()
{
    if (state_->position + offset_ > params_.max_position)
    {
        this->advance();
    }

    int idx = state_->position / 64;
    int offset = state_->position % 64;
    int num_bits = 64 - offset;

    RanluxppUInt bits = state_->value.number[idx] >> offset;
    if (num_bits < offset_)
    {
        bits |= state_->value.number[idx + 1] << num_bits;
    }
    bits &= ((RanluxppUInt(1) << offset_) - 1);

    state_->position += offset_;

    CELER_ENSURE(state_->position <= params_.max_position);
    return bits;
}

//---------------------------------------------------------------------------//
/*!
 * Produce the next block of random bits.
 */
CELER_FUNCTION void RanluxppRngEngine::advance()
{
    RanluxppArray9 lcg = celeritas::detail::to_lcg(state_->value);
    lcg = celeritas::detail::compute_mod_multiply(params_.advance_state, lcg);
    state_->value = celeritas::detail::to_ranlux(lcg);
    state_->position = 0;
}

//---------------------------------------------------------------------------//
}  // end namespace celeritas
