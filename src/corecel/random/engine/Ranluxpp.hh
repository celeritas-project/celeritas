//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/engine/Ranluxpp.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cassert>
#include <cstdint>

#include "detail/RanluxppLCG.hh"
#include "detail/RanluxppMulMod.hh"

namespace
{

__device__ const uint64_t kA_2048[] = {
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
//---------------------------------------------------------------------------//
/*!
 * Implementation of the RanluxPP RNG Engine
 *
 * \tparam w  Position offset amount in bits
 */
template<int w>
class RanluxppEngineImpl
{
  private:
    uint64_t fState_[9];  ///< RANLUX state of the generator
    unsigned fCarry_;  ///< Carry bit of the RANLUX state
    int fPosition_ = 0;  ///< Current position in bits

    static constexpr uint64_t const* kA_ = kA_2048;
    static constexpr int kMaxPos_ = 9 * 64;

  protected:
    CELER_FUNCTION void saveState(uint64_t* state) const
    {
        for (int i = 0; i < 9; ++i)
        {
            state[i] = fState_[i];
        }
    }

    CELER_FUNCTION void xorState(uint64_t const* state)
    {
        for (int i = 0; i < 9; ++i)
        {
            fState_[i] ^= state[i];
        }
    }

  public:
    RanluxppEngineImpl() = default;

    //! Produce next block of random bits
    CELER_FUNCTION void __attribute__((noinline)) advance()
    {
        uint64_t lcg[9];
        to_lcg(fState_, fCarry_, lcg);
        mulmod(kA_, lcg);
        to_ranlux(lcg, fState_, fCarry_);
        fPosition_ = 0;
    }

    //! Return the next random bits, generate a new block if necessary
    CELER_FUNCTION uint64_t nextRandomBits()
    {
        if (fPosition_ + w > kMaxPos_)
        {
            this->advance();
        }

        int idx = fPosition_ / 64;
        int offset = fPosition_ % 64;
        int numBits = 64 - offset;

        uint64_t bits = fState_[idx] >> offset;
        if (numBits < w)
        {
            bits |= fState_[idx + 1] << numBits;
        }
        bits &= ((uint64_t(1) << w) - 1);

        fPosition_ += w;
        CELER_ASSERT(fPosition_ <= kMaxPos_, "position out of range!");

        return bits;
    }

    //! Return a floating point number, converted from the next random bits.
    CELER_FUNCTION double nextRandomFloat()
    {
        static constexpr double div = 1.0 / (uint64_t(1) << w);
        uint64_t bits = this->nextRandomBits();
        return bits * div;
    }

    //! Initialize and seed the state of the generator
    CELER_FUNCTION void setSeed(uint64_t s)
    {
        uint64_t lcg[9];
        lcg[0] = 1;
        for (int i = 1; i < 9; ++i)
        {
            lcg[i] = 0;
        }

        uint64_t a_seed[9];
        // Skip 2 ** 96 states.
        powermod(kA_, a_seed, uint64_t(1) << 48);
        powermod(a_seed, a_seed, uint64_t(1) << 48);
        // Skip another s states.
        powermod(a_seed, a_seed, s);
        mulmod(a_seed, lcg);

        to_ranlux(lcg, fState_, fCarry_);
        fPosition_ = 0;
    }

    //! Skip `n` random numbers without generating them
    CELER_FUNCTION void skip(uint64_t n)
    {
        int left = (kMaxPos_ - fPosition_) / w;
        CELER_ASSERT(left >= 0, "position was out of range!");
        if (n < (uint64_t)left)
        {
            // Just skip the next few entries in the currently available bits.
            fPosition_ += n * w;
            assert(fPosition_ <= kMaxPos_, "position out of range!");
            return;
        }

        n -= left;
        // Need to advance and possibly skip over blocks.
        int nPerState = kMaxPos_ / w;
        int skip = (n / nPerState);

        uint64_t a_skip[9];
        powermod(kA_, a_skip, skip + 1);

        uint64_t lcg[9];
        to_lcg(fState_, fCarry_, lcg);
        mulmod(a_skip, lcg);
        to_ranlux(lcg, fState_, fCarry_);

        // Potentially skip numbers in the freshly generated block.
        int remaining = n - skip * nPerState;
        CELER_ASSERT(remaining >= 0,
                     "should not end up at a negative position!");
        fPosition_ = remaining * w;
        CELER_ASSERT(fPosition <= kMaxPos && "position out of range!");
    }
};

//---------------------------------------------------------------------------//

class RanluxppDouble final : public RanluxppEngineImpl<48>
{
  public:
    //! Instantiate with optional default seed
    CELER_FUNCTION RanluxppDouble(uint64_t seed = 314159265)
    {
        this->SetSeed(seed);
    }

    //! Generate a double-precision random number with 48 bits of randomness
    CELER_FUNCTION double rndm() { return (*this)(); }

    //! Generate a double-precision random number (non-virtual method)
    CELER_FUNCTION double operator()() { return this->NextRandomFloat(); }

    //! Generate a random integer value with 48 bits
    CELER_FUNCTION uint64_t intRndm() { return this->NextRandomBits(); }

    //! Generate a uniformly random 64-bit integer by concatenating two 32-bit
    //! words
    CELER_FUNCTION uint64_t intRndm64()
    {
        // draw two 48-bit words, but take only their low 32 bits each
        uint64_t lo = this->NextRandomBits() & 0xFFFFFFFFu;
        uint64_t hi = this->NextRandomBits() & 0xFFFFFFFFu;
        return (lo << 32) | hi;
    }

    //! Branch a new RNG state, also advancing the current one.
    //! The caller must Advance() the branched RNG state to decorrelate the
    //! produced numbers.
    CELER_FUNCTION RanluxppDouble branchNoAdvance()
    {
        // Save the current state, will be used to branch a new RNG.
        uint64_t oldState[9];
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
