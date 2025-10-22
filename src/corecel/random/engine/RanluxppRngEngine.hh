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
    inline CELER_FUNCTION RanluxppRngEngine& operator=(RanluxppUInt seed)
    {
        this->setSeed(seed);
        return *this;
    }

    //! Initialize and seed the state of the generator.
    CELER_FUNCTION void setSeed(RanluxppUInt seed)
    {
        celeritas::initialize_state(*state_, seed, params_);
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
    CELER_FUNCTION void skip(RanluxppUInt n);

    // Return the next random bits, generate a new block if necessary
    CELER_FUNCTION RanluxppUInt nextRandomBits();

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
        RanluxppArray9 lcg
            = celeritas::detail::to_lcg(state_->state, state_->carry);
        lcg = celeritas::detail::compute_mod_multiply(params_.kA_2048, lcg);
        state_->state = celeritas::detail::to_ranlux(lcg, state_->carry);
        state_->position = 0;
    }

    /// DATA ///
    static constexpr int offset_ = 48;
    ParamsRef const& params_;
    RanluxppRngState* state_;
};

//---------------------------------------------------------------------------//
}  // end namespace celeritas
