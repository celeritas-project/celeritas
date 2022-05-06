//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file XorwowRngData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Array.hh"
#include "base/Collection.hh"
#include "base/Macros.hh"
#include "base/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Persistent data for XORWOW generator.
 *
 * If we want to add the "discard" operation or support initialization with a
 * subsequence or offset, we can add the precomputed XORWOW jump matrices here.
 */
template<Ownership W, MemSpace M>
struct XorwowRngParamsData
{
    // TODO: 256-bit seed used to generate initial states for the RNGs
    // For now, just 4 bytes (same as our existing cuda/hip interface)
    Array<unsigned int, 1> seed;

    //// METHODS ////

    //! Whether the data is assigned
    explicit CELER_FUNCTION operator bool() const { return true; }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    XorwowRngParamsData& operator=(const XorwowRngParamsData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        seed = other.seed;
        return *this;
    }
};

//---------------------------------------------------------------------------//
//! Individual RNG state
struct XorwowState
{
    using uint_t = unsigned int;
    static_assert(sizeof(uint_t) == 4, "Expected 32-bit int");

    Array<uint_t, 5> xorstate;  //!< x, y, z, w, v
    uint_t           weylstate; //!< d
};

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

    StateItems<XorwowState> state; //!< Track state [track]

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
/*!
 * Resize and seed the RNG states.
 *
 * cuRAND and hipRAND implement functions that permute the original (but
 * seemingly arbitrary) seeds given in Marsaglia with 64 bits of seed data. We
 * simply generate pseudorandom, independent starting states for all data in
 * all threads using MT19937.
 */
template<MemSpace M>
void resize(
    XorwowRngStateData<Ownership::value, M>* state,
    const XorwowRngParamsData<Ownership::const_reference, MemSpace::host>& params,
    size_type size);

//---------------------------------------------------------------------------//
} // namespace celeritas
