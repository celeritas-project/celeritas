//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/data/RanluxppRngData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/random/data/RanluxppTypes.hh"

namespace celeritas
{

//---------------------------------------------------------------------------//
/*!
 * Persistent trivially copiable data for the Ranluxpp random number generator.
 */
struct RanluxppRngParamsDataImpl
{
    //// DATA ////

    //! User-provided seed
    RanluxppUInt seed = 0;

    //! Maximum bit position in the state
    static constexpr int max_position = sizeof(RanluxppArray9) * 8;

    //! Stores \f$a^2048 mod m\f$ for RCARRY values of \f$a\f$ and \f$m\f$.
    RanluxppArray9 advance_state;

    //! Stores \f$a^(2048 * (2^96)) mod m\f$
    RanluxppArray9 advance_sequence;

    //// FUNCTIONS ////
    //! Whether the data is assigned.
    explicit CELER_CONSTEXPR_FUNCTION operator bool() const
    {
        return advance_state[0] != 0 && advance_sequence[0] != 0;
    }
};

template<Ownership W, MemSpace M>
struct RanluxppRngParamsData : RanluxppRngParamsDataImpl
{
    //! Assign from another set of data.
    template<Ownership W2, MemSpace M2>
    RanluxppRngParamsData& operator=(RanluxppRngParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        static_cast<RanluxppRngParamsDataImpl&>(*this) = other;
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

//---------------------------------------------------------------------------//
/*!
 * Initializer object for the Ranluxpp engine
 */
struct RanluxppInitializer
{
    //! Seed
    RanluxppUInt seed{0ull};

    //! Thread-local id.
    RanluxppUInt subsequence{0ull};

    //! Offset into rng stream
    RanluxppUInt offset{0ull};
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
