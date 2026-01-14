//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/data/XorwowRngData.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstdint>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/Collection.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! 32-bit unsigned integer type for xorwow
using XorwowUInt = std::uint32_t;
//! Seed type used to generate initial states for the RNG
using XorwowSeed = Array<XorwowUInt, 1>;

//---------------------------------------------------------------------------//
/*!
 * Persistent data for XORWOW generator.
 */
template<Ownership W, MemSpace M>
struct XorwowRngParamsData
{
    //// TYPES ////

    using JumpPoly = Array<XorwowUInt, 5>;
    using ArrayJumpPoly = Array<JumpPoly, 32>;

    //// DATA ////

    //! \todo Use full 256-bit seed to generate initial states for the RNGs
    //! For now, just 4 bytes (same as our existing cuda/hip interface)
    XorwowSeed seed;

    // Jump polynomials
    ArrayJumpPoly jump;
    ArrayJumpPoly jump_subsequence;

    //// METHODS ////

    static CELER_CONSTEXPR_FUNCTION size_type num_words()
    {
        return JumpPoly{}.size();
    }
    static CELER_CONSTEXPR_FUNCTION size_type num_bits()
    {
        return 8 * sizeof(XorwowUInt);
    }

    //! Whether the data is assigned
    explicit CELER_FUNCTION operator bool() const { return true; }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    XorwowRngParamsData& operator=(XorwowRngParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        seed = other.seed;
        jump = other.jump;
        jump_subsequence = other.jump_subsequence;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Initialize an RNG.
 */
struct XorwowRngInitializer
{
    Array<unsigned int, 1> seed{0};
    ull_int subsequence{0};
    ull_int offset{0};
};

//---------------------------------------------------------------------------//
//! Individual RNG state
struct XorwowState
{
    Array<XorwowUInt, 5> xorstate;  //!< x, y, z, w, v
    XorwowUInt weylstate;  //!< d
};

//---------------------------------------------------------------------------//
//! Initializes an RNG state for a branched RNG
using XorwowRngStateInitializer = XorwowState;

//---------------------------------------------------------------------------//
/*!
 * XORWOW generator states for all threads.
 */
template<Ownership W, MemSpace M>
struct XorwowRngStateData
{
    //// TYPES ////

    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using StateItems = StateCollection<T, W, M>;

    //// DATA ////

    StateItems<XorwowState> state;  //!< Track state [track]

    //// METHODS ////

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const { return !state.empty(); }

    //! State size
    CELER_FUNCTION size_type size() const { return state.size(); }

    //! Assign from another set of states
    template<Ownership W2, MemSpace M2>
    XorwowRngStateData& operator=(XorwowRngStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        state = other.state;
        return *this;
    }
};

//---------------------------------------------------------------------------//
// Initialize XORWOW states with well-distributed random data
void initialize_xorwow(Span<XorwowState> state,
                       XorwowSeed const& seed,
                       StreamId stream);

//---------------------------------------------------------------------------//
// Resize and seed the RNG states
template<MemSpace M>
void resize(XorwowRngStateData<Ownership::value, M>* state,
            HostCRef<XorwowRngParamsData> const& params,
            StreamId stream,
            size_type size);

//---------------------------------------------------------------------------//
}  // namespace celeritas
