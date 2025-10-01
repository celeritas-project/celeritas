//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/data/RanluxppData.hh
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/Collection.hh"
#include "corecel/random/data/RanluxppTypes.hh"
#include "corecel/random/data/detail/RanluxppLCG.hh"
#include "corecel/random/data/detail/RanluxppMulMod.hh"

namespace celeritas
{

//---------------------------------------------------------------------------//
/*!
 * Persistent data for the Ranluxpp random number generator
 */
template<Ownership W, MemSpace M>
struct RanluxppRngParamsData
{
    //// DATA ////
    RanluxppArray9 kA_2048;
    RanluxppUInt kMaxPos;
    RanluxppUInt seed = 314159265;

    //// FUNCTIONS ////
    //! Whether the data is assigned
    explicit CELER_FUNCTION operator bool() const { return !kA_2048.empty(); }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    RanluxppRngParamsData<W, M>&
    operator=(RanluxppRngParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        kA_2048 = other.kA_2048;
        kMaxPos = other.kMaxPos;
        seed = other.seed;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Individual RNG state for Ranluxpp
 */
struct RanluxppRngState
{
    //// DATA ////
    RanluxppArray9 state;
    RanluxppUInt carry;
    int position;

    //! Pickle the state into the given array
    CELER_FUNCTION void saveState(RanluxppArray9& state) const
    {
        std::copy_n(state.cbegin(), 9, state.begin());
    }

    //! Perform XOR operation on state
    CELER_FUNCTION void xorState(RanluxppArray9 const& other_state)
    {
        std::transform(other_state.cbegin(),
                       other_state.cend(),
                       state.cbegin(),
                       state.begin(),
                       [](RanluxppUInt a, RanluxppUInt b) { return a ^ b; });
    }

    // Initialize the state with the given seed
    CELER_FUNCTION void
    initialize(RanluxppUInt seed, RanluxppArray9 const& kA_2048)
    {
        // Skip 2 ** 96 states
        RanluxppArray9 a_seed;
        celeritas::detail::powermod(kA_2048, a_seed, RanluxppUInt(1) << 48);
        celeritas::detail::powermod(a_seed, a_seed, RanluxppUInt(1) << 48);

        // Skip another s states.
        celeritas::detail::powermod(a_seed, a_seed, seed);
        RanluxppArray9 lcg = {0, 1, 1, 1, 1, 1, 1, 1, 1};
        celeritas::detail::mulmod(a_seed, lcg);

        // Set state and carry variable
        celeritas::detail::toRanlux(lcg, state, carry);
        position = 0;
    }

    // Produce the next block of random bits
    CELER_FUNCTION void advance(RanluxppArray9 const& kA)
    {
        RanluxppArray9 lcg;
        celeritas::detail::toLCG(state, carry, lcg);
        celeritas::detail::mulmod(kA, lcg);
        celeritas::detail::toRanlux(lcg, state, carry);
        position = 0;
    }

    // Skip 'n' random numbers without generating them
    CELER_FUNCTION void skip(RanluxppUInt n,
                             RanluxppUInt kMaxPos,
                             RanluxppArray9 const& kA_2048,
                             RanluxppUInt offset)
    {
        CELER_ASSERT(n > 0);
        CELER_ASSERT(kMaxPos > 0);

        int left = (kMaxPos - position) / offset;
        CELER_ASSERT(left >= 0);
        if (n < static_cast<RanluxppUInt>(left))
        {
            // Just skip the next few entries in the currently
            // available bits.
            position += n * offset;
            CELER_ASSERT(position <= kMaxPos);
            return;
        }

        n -= left;
        // Need to advance and possibly skip over blocks.
        int nPerState = kMaxPos / offset;
        int skip = (n / nPerState);

        RanluxppArray9 a_skip;
        celeritas::detail::powermod(kA_2048, a_skip, skip + 1);

        RanluxppArray9 lcg;
        celeritas::detail::toLCG(state, carry, lcg);
        celeritas::detail::mulmod(a_skip, lcg);
        celeritas::detail::toRanlux(lcg, state, carry);

        // Potentially skip numbers in the freshly generated block.
        int remaining = n - skip * nPerState;
        CELER_ASSERT(remaining >= 0);
        position = remaining * offset;
        CELER_ASSERT(position <= kMaxPos);
    }
};

//---------------------------------------------------------------------------//
/*!
 * State data for Ranluxpp generator
 */
template<Ownership W, MemSpace M>
struct RanluxppRngStateData
{
    //// TYPES ////
    template<class T>
    using StateItems = StateCollection<T, W, M>;

    //// DATA ////

    StateItems<RanluxppRngState> state;

    //// METHODS ////
    // Pickle the current state into the given array
    inline CELER_FUNCTION void saveState(RanluxppArray9& state) const;

    // XOR the state with the given state array
    inline CELER_FUNCTION void xorState(RanluxppArray9 const& state) const;

    // Advance the state
    inline CELER_FUNCTION void advance(RanluxppArray9 const& kA);
};

//---------------------------------------------------------------------------//
// Initialize Ranluxpp states with well-distributed random data
void initialize_ranluxpp(Span<RanluxppRngState> state,
                         RanluxppUInt const& seed,
                         StreamId stream);

//---------------------------------------------------------------------------//
// Resize and seed the RNG states
template<MemSpace M>
void resize(RanluxppStateData<Ownership::value, M>* state,
            HostCRef<RanluxppRngParamsData> const& params,
            StreamId stream,
            size_type size);

//---------------------------------------------------------------------------//
}  // namespace celeritas
