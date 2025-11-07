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
 * Persistent data for the Ranluxpp random number generator.
 */
template<Ownership W, MemSpace M>
struct RanluxppRngParamsData
{
    //// DATA ////
    RanluxppUInt seed = 0;
    int max_position = 0;
    RanluxppArray9 state_2048;
    RanluxppArray9 seed_state;

    //// FUNCTIONS ////
    //! Whether the data is assigned.
    explicit CELER_FUNCTION operator bool() const
    {
        return !state_2048.empty() && !seed_state.empty();
    }

    //! Assign from another set of data.
    template<Ownership W2, MemSpace M2>
    RanluxppRngParamsData& operator=(RanluxppRngParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        seed = other.seed;
        max_position = other.max_position;
        state_2048 = other.state_2048;
        seed_state = other.seed_state;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Individual RNG state for Ranluxpp.
 */
struct RanluxppRngState
{
    //// DATA ////
    //! Ranluxpp state number and carry bit
    RanluxppNumber value;
    //! Current position in the state.
    int position;
};

struct RanluxppInitializer
{
    //! Thread-local id.
    RanluxppUInt thread_local_id;
};

//---------------------------------------------------------------------------//
/*!
 * State data for Ranluxpp generator.
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
    //! True if assigned.
    explicit CELER_FUNCTION operator bool() const { return !state.empty(); }

    //! State size.
    CELER_FUNCTION size_type size() const { return state.size(); }

    //! Assign from another set of states.
    template<Ownership W2, MemSpace M2>
    RanluxppRngStateData& operator=(RanluxppRngStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        state = other.state;
        return *this;
    }
};

//---------------------------------------------------------------------------//
// Resize and seed the RNG states
template<MemSpace M>
void resize(RanluxppRngStateData<Ownership::value, M>* state,
            HostCRef<RanluxppRngParamsData> const& params,
            StreamId stream,
            size_type size);

//---------------------------------------------------------------------------//
}  // namespace celeritas
