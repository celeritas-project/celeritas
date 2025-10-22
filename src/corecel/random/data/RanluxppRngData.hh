//------------------------------- -*- C++ -*- -------------------------------//
// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0
//---------------------------------------------------------------------------//
/*!
 * \file corecel/random/data/RanluxppRngData.hh
 *
 * Original source:
 * https://github.com/apt-sim/AdePT/blob/master/include/AdePT/copcore/Ranluxpp.h
 */
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/Collection.hh"
#include "corecel/random/data/RanluxppTypes.hh"
#include "corecel/random/data/detail/RanluxppImpl.hh"

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
    int kMaxPos;
    RanluxppUInt seed = 314159265;

    //// FUNCTIONS ////
    //! Whether the data is assigned
    explicit CELER_FUNCTION operator bool() const { return !kA_2048.empty(); }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    RanluxppRngParamsData& operator=(RanluxppRngParamsData<W2, M2> const& other)
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
    unsigned int carry;
    int position;

    //! Pickle the state into the given array
    CELER_FUNCTION void saveState(RanluxppArray9& state) const
    {
        std::copy_n(state.cbegin(), 9, state.begin());
    }

    //! Perform XOR operation on state
    CELER_FUNCTION void xorState(RanluxppArray9 const& other_state)
    {
        for (auto i : celeritas::range(9))
        {
            state[i] ^= other_state[i];
        }
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
    //! True if assigned
    explicit CELER_FUNCTION operator bool() const { return !state.empty(); }

    //! State size
    CELER_FUNCTION size_type size() const { return state.size(); }

    //! Assign from another set of states
    template<Ownership W2, MemSpace M2>
    RanluxppRngStateData& operator=(RanluxppRngStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        state = other.state;
        return *this;
    }
};

//---------------------------------------------------------------------------//
// Initialize a single Ranluxpp state
void initialize_state(RanluxppRngState& state,
                      RanluxppUInt seed,
                      HostCRef<RanluxppRngParamsData> const& params);

//---------------------------------------------------------------------------//
// Initialize Ranluxpp states with well-distributed random data
void initialize_ranluxpp(Span<RanluxppRngState> state,
                         HostCRef<RanluxppRngParamsData> const& params,
                         StreamId stream);

//---------------------------------------------------------------------------//
// Resize and seed the RNG states
template<MemSpace M>
void resize(RanluxppRngStateData<Ownership::value, M>* state,
            HostCRef<RanluxppRngParamsData> const& params,
            StreamId stream,
            size_type size);

//---------------------------------------------------------------------------//
}  // namespace celeritas
