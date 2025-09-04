//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/data/RanluxppData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/Collection.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! 64-bit unsigned integer type for Ranluxpp
using RanluxppUInt = std::uint64_t;
//! State array for Ranluxpp
using RanluxppStateArray = Array<RanluxppUInt, 9>;

//---------------------------------------------------------------------------//
/*!
 * Persistent data for Ranluxpp generator
 */
template<Ownership W, Memspace M>
struct RanluxppParamsData
{
    //// DATA ////
    RanluxppStateArray const kA_2048_ = {
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
    RanluxppUInt const kMaxPos_;
};

//---------------------------------------------------------------------------//
/*!
 * Individual RNG state for Ranluxpp
 */
struct RanluxppState
{
    RanluxppStateArray fstate_;
    RanluxppUInt fCarry_;
    int fPosition_;
};

//---------------------------------------------------------------------------//
/*!
 * State data for Ranluxpp generator
 */
template<Ownership W, Memspace M>
struct RanluxppStateData
{
    //// TYPES ////
    template<class T>
    using StateItems = StateCollection<T, W, M>;

    //// DATA ////

    StateItems<RanluxppState> state;

    //// METHODS ////
    // Pickle the current state into the given array
    inline CELER_FUNCTION void saveState(ArrayUInt state) const;

    // XOR the state with the given state array
    inline CELER_FUNCTION void xorState(ArrayUInt const& state) const;

    // Advance the state
    inline CELER_FUNCTION void advance(ArrayUInt const& kA);

    ArrayUInt fState_;
    RanluxppUInt fCarry_;
    int fPosition_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with defaults.
 */
RanluxppData::RanluxppData() {}

//---------------------------------------------------------------------------//
/*!
 * Pickle the current state into the given array
 *
 * \param[out] state  The array to save the state into
 */
template<Ownership W, Memspace M>
inline CELER_FUNCTION void
RanluxppStateData<W, M>::saveState(ArrayUint state) const
{
    for (int i : celeritas::range(9))
    {
        state[i] = fState_[i];
    }
}

//---------------------------------------------------------------------------//
/*!
 * XOR the state with the given state array
 *
 * \param[in] state  The state array to XOR with the current state vector
 */
template<Ownership W, Memspace M>
inline CELER_FUNCTION void
RanluxppStateData<W, M>::xorState(ArrayUInt const state) const
{
    for (int i : celeritas::range(9))
    {
        fState_[i] ^= state[i];
    }
}

//---------------------------------------------------------------------------//
/*!
 * Produce next block of random bits
 */
template<Ownership W, Memspace M>
inline CELER_FUNCTION void
RanluxppStateData<W, M>::advance(RanluxppStateArray kA)
{
    ArrayUInt lcg;
    celeritas::detail::toLCG(fState_, fCarry_, lcg);
    celeritas::detail::mulmod(kA, lcg);
    celeritas::detail::toRanlux(lcg, fState_, fCarry_);
    fPosition_ = 0;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
