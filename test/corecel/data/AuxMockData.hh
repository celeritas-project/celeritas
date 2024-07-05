//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/AuxMockData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Shared diagnostic attributes.
 */
template<Ownership W, MemSpace M>
struct AuxMockParamsData
{
    //// TYPES ////

    template<class T>
    using Items = Collection<T, W, M>;

    //// DATA ////

    //! Number of tally bins
    size_type num_bins{0};

    //! Integer values
    Items<int> integers;

    //// METHODS ////

    //! Whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return num_bins > 0 && !integers.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    AuxMockParamsData& operator=(AuxMockParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        num_bins = other.num_bins;
        integers = other.integers;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * State data for accumulating results for each particle type.
 *
 * \c counts is indexed as particle_id * num_bins + bin_index.
 */
template<Ownership W, MemSpace M>
struct AuxMockStateData
{
    //// TYPES ////

    template<class T>
    using Items = Collection<T, W, M>;

    template<class T>
    using StateItems = StateCollection<T, W, M>;

    //// DATA ////

    StreamId stream;
    StateItems<int> local_state;
    Items<size_type> counts;

    //// METHODS ////

    //! Number of states
    CELER_FUNCTION size_type size() const { return local_state.size(); }

    //! Whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return stream && !local_state.empty() && !counts.empty();
    }

    //! Assign from another set of states
    template<Ownership W2, MemSpace M2>
    AuxMockStateData& operator=(AuxMockStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        stream = other.stream;
        local_state = other.local_state;
        counts = other.counts;
        return *this;
    }
};

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//

template<MemSpace M>
inline void resize(AuxMockStateData<Ownership::value, M>* state,
                   HostCRef<AuxMockParamsData> const& params,
                   StreamId sid,
                   size_type count)
{
    CELER_EXPECT(params);
    state->stream = sid;
    resize(&state->local_state, count);
    resize(&state->counts, params.num_bins);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
