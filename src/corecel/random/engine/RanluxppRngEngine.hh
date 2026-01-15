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
#include "corecel/random/distribution/GenerateCanonical.hh"
#include "corecel/random/distribution/detail/GenerateCanonical32.hh"
#include "corecel/sys/ThreadId.hh"

#include "detail/RanluxppImpl.hh"

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
 * bits of each is used as entropy.
 *
 * \todo The decision to discard the high bits rather than the low bits from
 * each word is likely undesirable at least for the last number in the state,
 * since it is known that LCG integers have highly correlated sequential LSBs.
 * Also instead, the class could be adapted to provide 18 32-bit samples
 * instead of discarding 16 out of every 48 bits.
 */
class RanluxppRngEngine
{
  public:
    //@{
    //! Public types.
    using result_type = std::uint32_t;
    using Initializer_t = RanluxppInitializer;
    using ParamsRef = NativeCRef<RanluxppRngParamsData>;
    using StateRef = NativeRef<RanluxppRngStateData>;
    using RngStateInitializer_t = RanluxppRngStateInitializer;
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
        return celeritas::numeric_limits<result_type>::max();
    }

    // Initialize state with the given seed initializer
    inline CELER_FUNCTION RanluxppRngEngine&
    operator=(Initializer_t const& init);

    // Initialize state with the given state initializer
    inline CELER_FUNCTION RanluxppRngEngine&
    operator=(RngStateInitializer_t const& state_init);

    // Generate a 32-bit random integer.
    inline CELER_FUNCTION result_type operator()();

    // Advance the state \c count times.
    inline CELER_FUNCTION void discard(RanluxppUInt count);

    // Initialize a state for a new spawned RNG.
    inline CELER_FUNCTION RngStateInitializer_t branch();

  private:
    /// IMPLEMENTATION ///

    // Produce the next block of random bits for the given state.
    inline CELER_FUNCTION void advance(RanluxppRngState& state);

    /// DATA ///
    static constexpr int offset_ = 48;
    ParamsRef const& params_;
    RanluxppRngState* state_;
};

//---------------------------------------------------------------------------//
/*!
 * Specialization of GenerateCanonical for RanluxppRngEngine.
 */
template<class RealType>
struct GenerateCanonical<RanluxppRngEngine, RealType>
{
    //!@{
    //! \name Type aliases
    using real_type = RealType;
    using result_type = RealType;
    //!@}

    //! Declare that we use the 32-bit canonical generator
    static constexpr auto policy = GenerateCanonicalPolicy::builtin32;

    //! Sample a random number on [0, 1)
    CELER_FORCEINLINE_FUNCTION result_type operator()(RanluxppRngEngine& rng)
    {
        return detail::GenerateCanonical32<RealType>()(rng);
    }
};

//---------------------------------------------------------------------------//
// INLINE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Initialize state for the given seed and subsequence.
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
 * Initialize state for the given state initializer.
 */
inline CELER_FUNCTION RanluxppRngEngine&
RanluxppRngEngine::operator=(RngStateInitializer_t const& state_init)
{
    (*state_).value = state_init.value;
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Skip `n` random numbers without generating them.
 */
CELER_FUNCTION void RanluxppRngEngine::discard(RanluxppUInt n)
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
 * Return the next random bits, generating a new block if necessary.
 */
CELER_FUNCTION auto RanluxppRngEngine::operator()() -> result_type
{
    if (state_->position + offset_ > params_.max_position)
    {
        this->advance(*state_);
    }

    // Extract the Nth word from the state
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
    return bits & 0xffffffffu;
}

//---------------------------------------------------------------------------//
/*!
 * Initialize a state for a new spawned RNG.
 *
 * \par Branching
 * Branching is performed in two steps.  First, the state of the new RNG
 * (\f$x^{\prime}\f$) is initialized as
 * \f[
 *      x^{i,\prime}_j = x^i_j ^ x^{i+1}_j \, .
 * \f]
 * Second, to decorrelate the new RNG from this RNG, the new RNG is
 * advanced forward to the next block
 */
CELER_FUNCTION RanluxppRngEngine::RngStateInitializer_t
RanluxppRngEngine::branch()
{
    // Create a new state
    RanluxppRngState new_state = *state_;

    // Advance the RNG
    this->advance(*state_);

    // XOR the new state with the updated and advanced state
    for (auto i : celeritas::range(9))
    {
        new_state.value.number[i] ^= (*state_).value.number[i];
    }

    // Advance the new rng to decorrelate it
    this->advance(new_state);

    // Create and return the state initializer
    RngStateInitializer_t initializer{new_state.value};

    return initializer;
}

//---------------------------------------------------------------------------//
/*!
 * Advance to the next state (block of random bits).
 */
CELER_FUNCTION void RanluxppRngEngine::advance(RanluxppRngState& state)
{
    RanluxppArray9 lcg = celeritas::detail::to_lcg(state.value);
    lcg = celeritas::detail::compute_mod_multiply(params_.advance_state, lcg);
    state.value = celeritas::detail::to_ranlux(lcg);
    state.position = 0;
}

//---------------------------------------------------------------------------//
}  // end namespace celeritas
