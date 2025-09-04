//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/engine/Ranluxpp.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstdint>
#include <string>

#include "corecel/Assert.hh"
#include "corecel/random/data/detail/RanluxppLCG.hh"
#include "corecel/random/data/detail/RanluxppMulMod.hh"

// ******** Temporary
#include <iostream>

#include "corecel/cont/ArrayIO.hh"

namespace
{

uint64_t const kA_2048[] = {
    0xed7faa90747aaad9,
    0x4cec2c78af55c101,
    0xe64dcb31c48228ec,
    0x6d8a15a13bee7cb0,
    0x20b2ca60cb78c509,
    0x256c3d3c662ea36c,
    0xff74e54107684ed2,
    0x492edfcc0cc8e753,
    0xb48c187cf5b22097,
};

//---------------------------------------------------------------------------//
}  // end anonymous namespace

namespace celeritas
{
using RanluxppUInt = std::uint64_t;
using RanluxppStateArray = Array<RanluxppUInt, 9>;

//---------------------------------------------------------------------------//
/*!
 * Implementation of the RanluxPP RNG Engine
 *
 * \tparam w  Position offset amount in bits
 */
template<int w>
class RanluxppEngineImpl
{
  public:
    using State = RanluxppUInt*;

  private:
    RanluxppStateArray fState_;  ///< RANLUX state of the generator
    RanluxppUInt fCarry_;  ///< Carry bit of the RANLUX state
    int fPosition_ = 0;  ///< Current position in bits

    static constexpr RanluxppStateArray kA_ = {
        0xed7faa90747aaad9,
        0x4cec2c78af55c101,
        0xe64dcb31c48228ec,
        0x6d8a15a13bee7cb0,
        0x20b2ca60cb78c509,
        0x256c3d3c662ea36c,
        0xff74e54107684ed2,
        0x492edfcc0cc8e753,
        0xb48c187cf5b22097,
    };
    static constexpr int kMaxPos_ = 9 * 64;

  protected:
    CELER_FUNCTION void saveState(RanluxppStateArray state) const
    {
        for (int i : celeritas::range(9))
        {
            state[i] = fState_[i];
        }
    }

    CELER_FUNCTION void xorState(RanluxppStateArray const& state)
    {
        for (int i : celeritas::range(9))
        {
            fState_[i] ^= state[i];
        }
    }

  public:
    RanluxppEngineImpl() = default;

    //! Produce next block of random bits
    CELER_FUNCTION void __attribute__((noinline)) advance()
    {
        RanluxppStateArray lcg;
        celeritas::detail::toLCG(fState_, fCarry_, lcg);
        celeritas::detail::mulmod(kA_, lcg);
        celeritas::detail::toRanlux(lcg, fState_, fCarry_);
        fPosition_ = 0;
    }

    //! Return the next random bits, generate a new block if necessary
    CELER_FUNCTION RanluxppUInt nextRandomBits()
    {
        if (fPosition_ + w > kMaxPos_)
        {
            this->advance();
        }

        int idx = fPosition_ / 64;
        int offset = fPosition_ % 64;
        int numBits = 64 - offset;

        RanluxppUInt bits = fState_[idx] >> offset;
        if (numBits < w)
        {
            bits |= fState_[idx + 1] << numBits;
        }
        bits &= ((RanluxppUInt(1) << w) - 1);

        fPosition_ += w;
        CELER_ASSERT(fPosition_ <= kMaxPos_);

        return bits;
    }

    //! Return a floating point number, converted from the next random bits.
    CELER_FUNCTION double nextRandomFloat()
    {
        static constexpr double div = 1.0 / (RanluxppUInt(1) << w);

        RanluxppUInt bits = this->nextRandomBits();
        return bits * div;
    }

    //! Initialize and seed the state of the generator
    CELER_FUNCTION void setSeed(RanluxppUInt s)
    {
        RanluxppStateArray lcg;
        lcg[0] = 1;
        for (int i : celeritas::range(1, 9))
        {
            lcg[i] = 0;
        }

        RanluxppStateArray a_seed;
        // Skip 2 ** 96 states.
        celeritas::detail::powermod(kA_, a_seed, RanluxppUInt(1) << 48);
        celeritas::detail::powermod(a_seed, a_seed, RanluxppUInt(1) << 48);
        // Skip another s states.
        celeritas::detail::powermod(a_seed, a_seed, s);
        celeritas::detail::mulmod(a_seed, lcg);

        celeritas::detail::toRanlux(lcg, fState_, fCarry_);
        fPosition_ = 0;
    }

    //! Skip `n` random numbers without generating them
    CELER_FUNCTION void skip(RanluxppUInt n)
    {
        int left = (kMaxPos_ - fPosition_) / w;
        CELER_ASSERT(left >= 0);
        if (n < (RanluxppUInt)left)
        {
            // Just skip the next few entries in the currently available bits.
            fPosition_ += n * w;
            CELER_ASSERT(fPosition_ <= kMaxPos_);
            return;
        }

        n -= left;
        // Need to advance and possibly skip over blocks.
        int nPerState = kMaxPos_ / w;
        int skip = (n / nPerState);

        RanluxppStateArray a_skip;
        celeritas::detail::powermod(kA_, a_skip, skip + 1);

        RanluxppStateArray lcg;
        celeritas::detail::toLCG(fState_, fCarry_, lcg);
        celeritas::detail::mulmod(a_skip, lcg);
        celeritas::detail::toRanlux(lcg, fState_, fCarry_);

        // Potentially skip numbers in the freshly generated block.
        int remaining = n - skip * nPerState;
        CELER_ASSERT(remaining >= 0);
        fPosition_ = remaining * w;
        CELER_ASSERT(fPosition_ <= kMaxPos_ && "position out of range!");
    }
};

//---------------------------------------------------------------------------//

class RanluxppDouble final : public RanluxppEngineImpl<48>
{
    using Base = RanluxppEngineImpl<48>;

  public:
    using result_type = RanluxppUInt;

    //! Instantiate with optional default seed
    CELER_FUNCTION RanluxppDouble(RanluxppUInt seed = 314159265)
    {
        Base::setSeed(seed);
    }

    //! Lowest value potentially generated (check this)
    static CELER_CONSTEXPR_FUNCTION result_type min() { return 0u; }
    //! Highest value potentially generated (check this)
    static CELER_CONSTEXPR_FUNCTION result_type max()
    {
        // todo: use celeritas limits
        return std::numeric_limits<RanluxppUInt>::max();
    }

    //! Generate a double-precision random number with 48 bits of randomness
    // CELER_FUNCTION double rndm() { return (*this)(); }

    //! Generate a double-precision random number (non-virtual method)
    CELER_FUNCTION RanluxppUInt operator()() { return this->intRndm64(); }

    //! Generate a random integer value with 48 bits
    // CELER_FUNCTION RanluxppUInt intRndm() { return this->nextRandomBits(); }

    //! Generate a uniformly random 64-bit integer by concatenating two 32-bit
    //! words
    CELER_FUNCTION RanluxppUInt intRndm64()
    {
        // draw two 48-bit words, but take only their low 32 bits each
        RanluxppUInt lo = this->nextRandomBits() & 0xFFFFFFFFu;
        RanluxppUInt hi = this->nextRandomBits() & 0xFFFFFFFFu;
        // std::cout << "Lo/Hi: " << lo << "/" << hi << std::endl;
        return (lo << 32) | hi;
    }

    //! Branch a new RNG state, also advancing the current one.
    //! The caller must Advance() the branched RNG state to decorrelate the
    //! produced numbers.
    CELER_FUNCTION RanluxppDouble branchNoAdvance()
    {
        // Save the current state, will be used to branch a new RNG.
        RanluxppStateArray oldState;
        this->saveState(oldState);
        this->advance();

        // Copy and modify the new RNG state.
        RanluxppDouble newRNG(*this);
        newRNG.xorState(oldState);
        return newRNG;
    }

    /// Branch a new RNG state, also advancing the current one.
    CELER_FUNCTION RanluxppDouble branch()
    {
        RanluxppDouble newRNG(this->branchNoAdvance());
        newRNG.advance();
        return newRNG;
    }
};

//---------------------------------------------------------------------------//
}  // end namespace celeritas
